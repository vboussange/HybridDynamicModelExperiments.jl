using ComponentArrays
using Optimisers
using ADTypes
using ConcreteStructs: @concrete
import HybridModelling: SegmentedTimeSeries

function _default_callback(l, epoch, ts)
    if epoch % 10 == 0
        @info "Epoch $epoch: Loss = $l"
    end
end

@concrete struct LuxBackend <: AbstractOptimBackend
    opt::Optimisers.AbstractRule
    n_epochs::Int
    adtype::ADTypes.AbstractADType
    loss_fn::Any
    callback::Any
end

LuxBackend(opt, n_epochs, adtype, loss_fn) = LuxBackend(opt, n_epochs, adtype, loss_fn, _default_callback)

nameof(::LuxBackend) = "LuxBackend"


function _feature_wrapper((token, tsteps_batch))
    return [(; u0 = token[i],
                saveat = tsteps_batch[:, i],
                tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
            )
            for i in eachindex(token)]
end

function _get_ics(dataloader, infer_ics::InferICs)
    function _fun(tok)
        segment_data, _ = dataloader[tok]
        u0 = segment_data[:, 1]
        ParameterLayer(;constraint = infer_ics.u0_constraint,
                    init_value = (; u0))
    end
    ics_list = [ _fun(tok) for tok in tokens(dataloader)]
    return InitialConditions(ics_list)
end

# TODO: this function is type unstable
function train(backend::LuxBackend,
        model::AbstractLuxLayer,
        dataloader::SegmentedTimeSeries,
        experimental_setup::InferICs,
        rng = Random.default_rng(),
        luxtype = Lux.f64)

    dataloader = luxtype(dataloader)
    dataloader = tokenize(dataloader)

    ics = _get_ics(dataloader, experimental_setup)

    if !istrue(experimental_setup)
        ics = Lux.Experimental.FrozenLayer(ics)
    end

    ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(_feature_wrapper),
        initial_conditions = ics, model = model)

    ps, st = luxtype(Lux.setup(rng, ode_model_with_ics))
    ps = ComponentArray(ps) # We transforms ps to support all sensealg from SciMLSensitivity

    train_state = Training.TrainState(ode_model_with_ics, ps, st, backend.opt)
    best_ps = ps
    best_st = st
    best_loss = luxtype(Inf)
    for epoch in 1:(backend.n_epochs)
        tot_loss = luxtype(0.0) 
        for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader
            _, loss, _, train_state = Training.single_train_step!(
                backend.adtype,
                backend.loss_fn,
                ((batched_tokens, batched_tsteps), batched_segments),
                train_state)
            tot_loss += loss
        end
        if tot_loss < best_loss
            best_ps = get_parameter_values(train_state)
            best_st = get_state_values(train_state)
            best_loss = tot_loss
        end
        backend.callback(tot_loss, epoch, train_state)
    end
    segment_ics = []
    for i in tokens(dataloader)
        _, segment_tsteps = dataloader[i]
        t0 = segment_tsteps[1]
        push!(segment_ics, merge(ics((; u0 = i), best_ps.initial_conditions, best_st.initial_conditions)[1], (; t0)))
    end
    segment_ics = vcat(segment_ics...)

    return (; ps = best_ps.model, st = best_st.model, ics = segment_ics)
end

get_parameter_values(train_state::Training.TrainState) = train_state.parameters
get_state_values(train_state::Training.TrainState) = train_state.states
