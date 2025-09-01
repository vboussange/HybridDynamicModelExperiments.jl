using ComponentArrays
using Optimisers
using ADTypes
using ConcreteStructs: @concrete
import HybridModelling: SegmentedTimeSeries

@concrete struct LuxBackend <: AbstractOptimBackend
    opt::Optimisers.AbstractRule
    n_epochs::Int
    adtype::ADTypes.AbstractADType
    loss_fn::Any
    verbose_frequency::Any
    callback::Any
end

nameof(::LuxBackend) = "LuxBackend"

function LuxBackend(opt, n_epochs, adtype, loss_fn; verbose_frequency = 10,
        callback = (l, m, p, s) -> nothing)
    return LuxBackend(opt, n_epochs, adtype, loss_fn, verbose_frequency, callback)
end

# TODO: convert dataloader data to luxtype
# TODO: probably has type instability
function train(backend::LuxBackend,
        model::AbstractLuxLayer,
        dataloader::SegmentedTimeSeries,
        experimental_setup::InferICs,
        rng = Random.default_rng();
        luxtype = Lux.f64)

    # TODO: maybe that dataloader |> luxtype could work; otherwise, use fmap
    dataloader = SegmentedTimeSeries(
        luxtype(dataloader.data), dataloader.segmentsize, dataloader.shift,
        dataloader.batchsize, dataloader.nsegments, dataloader.shuffle,
        dataloader.partial_segment, dataloader.partial_batch,
        dataloader.indices, dataloader.imax, dataloader.rng)
    dataloader = tokenize(dataloader)

    ic_list = ParameterLayer[]
    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        t0 = segment_tsteps[1]
        if istrue(experimental_setup)
            push!(ic_list,
                ParameterLayer(constraint = experimental_setup.u0_constraint,
                    init_value = (; u0), init_state_value = (; t0)))
        else
            push!(ic_list, ParameterLayer(init_value = (;), init_state_value = (; t0, u0)))
        end
    end
    ics = InitialConditions(ic_list)

    function feature_wrapper((token, tsteps_batch))
        return [(; u0 = token[i],
                    saveat = tsteps_batch[:, i],
                    tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
                )
                for i in eachindex(token)]
    end

    ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(feature_wrapper),
        initial_conditions = ics, model = model)

    ps, st = Lux.setup(rng, ode_model_with_ics)
    ps = ps |> luxtype |> ComponentArray # We transforms ps to support all sensealg from SciMLSensitivity

    train_state = Training.TrainState(ode_model_with_ics, ps, st, backend.opt)
    best_ps = ps
    best_st = st
    best_loss = Inf |> luxtype
    info = []
    for epoch in 1:(backend.n_epochs)
        tot_loss = 0.0 |> luxtype
        for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader
            _, loss, _, train_state = Training.single_train_step!(
                backend.adtype,
                backend.loss_fn,
                ((batched_tokens, batched_tsteps), batched_segments),
                train_state)
            tot_loss += loss
        end
        if epoch % backend.verbose_frequency == 0
            println("Epoch $epoch: Total Loss = ", tot_loss)
        end
        if tot_loss < best_loss
            best_ps = get_parameter_values(train_state)
            best_st = get_state_values(train_state)
            best_loss = tot_loss
        end
        push!(info,
            backend.callback(
                tot_loss, ode_model_with_ics, get_parameter_values(train_state),
                get_state_values(train_state)))
    end
    segment_ics, _ = ics([(; u0 = i) for i in tokens(dataloader)],
        best_ps.initial_conditions, best_st.initial_conditions)

    return (; ps = best_ps.model, st = best_st.model, ics = segment_ics, info)
end

get_parameter_values(train_state::Training.TrainState) = train_state.parameters
get_state_values(train_state::Training.TrainState) = train_state.states
