# TODO: For both InferICs{false} and InferICs{true}, we should define an InitialConditions layer, where in one case, parameters are null
# TODO: train could return an info object defined by callback, see https://turinglang.org/docs/tutorials/variational-inference/
using ComponentArrays

function train(::LuxBackend, 
                experimental_setup::InferICs;
                model::AbstractLuxLayer,
                rng=Random.default_rng(),
                dataloader,
                loss_fn = MSELoss(),
                adtype = Lux.AutoZygote(),
                verbose_frequency = 10,
                opt, 
                n_epochs, 
                callback = (l, m, p, s) -> nothing,
                u0_constraint = NoConstraint(),
                luxtype = Lux.f64,
                kwargs...)

    dataloader = tokenize(dataloader)

    ic_list = ParameterLayer[]
    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        t0 = segment_tsteps[1]
        if isa(experimental_setup, InferICs{false})
            push!(ic_list, ParameterLayer(constraint = u0_constraint, init_value = (;), init_state_value = (;t0, u0)))
        elseif isa(experimental_setup, InferICs{true})
            push!(ic_list, ParameterLayer(constraint = u0_constraint, init_value = (;u0), init_state_value = (;t0)))
        end
    end
    ics = InitialConditions(ic_list)

    function feature_wrapper((token, tsteps_batch))
        return [
            (;u0 = token[i],
            saveat = tsteps_batch[:, i], 
            tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
            )
            for i in 1:length(token)
        ]
    end

    ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(feature_wrapper), initial_conditions = ics, model = model)

    ps, st = Lux.setup(rng, ode_model_with_ics)
    ps = ps |> luxtype |> ComponentArray # We transforms ps to support all sensealg types
    train_state = Training.TrainState(ode_model_with_ics, ps, st, opt)
    best_ps = ps
    best_loss = Inf
    info = []
    for epoch in 1:n_epochs
        tot_loss = 0.
        for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader
            _, loss, _, train_state = Training.single_train_step!(
                adtype, 
                loss_fn, 
                ((batched_tokens, batched_tsteps), batched_segments),
                train_state)
            tot_loss += loss
        end
        if epoch % verbose_frequency == 0
            println("Epoch $epoch: Total Loss = ", tot_loss)
        end
        if tot_loss < best_loss
            best_ps = get_parameter_values(train_state)
            best_loss = tot_loss
        end
        push!(info, callback(tot_loss, ode_model_with_ics, get_parameter_values(train_state), get_state_values(train_state)))
    end
    best_model = StatefulLuxLayer{true}(ode_model_with_ics, best_ps, st)
    return (;best_model, info)
end

get_parameter_values(train_state::Training.TrainState) = train_state.parameters
get_state_values(train_state::Training.TrainState) = train_state.states
