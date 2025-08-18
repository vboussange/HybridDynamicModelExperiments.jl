function train(::LuxBackend, 
                ::InferICs{false};
                model::AbstractLuxLayer,
                rng,
                dataloader,
                loss_fn = LogMSELoss(),
                ad_backend = Lux.AutoZygote(),
                verbose_frequency = 10,
                opt = Adam(1e-3), 
                n_epochs = 1000, 
                kwargs...)

    function feature_wrapper((batched_segments, tsteps_batch))
        return [
            (;u0 = batched_segments[:, 1, i],
            saveat = tsteps_batch[:, i], 
            tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
            )
            for i in 1:batchsize
        ]
    end

    mychain = Chain(wrapper = Lux.WrappedFunction(feature_wrapper), model = model)
    ps, st = Lux.setup(rng, mychain)

    train_state = Training.TrainState(mychain, ps, st, opt)

    for epoch in 1:n_epochs
        tot_loss = 0.
        for (batched_segments, batched_tsteps) in dataloader
            _, loss, _, train_state = Training.single_train_step!(
                ad_backend,
                loss_fn, 
                ((batched_segments, batched_tsteps), batched_segments),
                train_state)
            tot_loss += loss
        end
        if epoch % verbose_frequency == 0
            println("Epoch $epoch: Total Loss = ", tot_loss)
        end
    end
    return train_state
end


function train(::LuxBackend, 
                ::InferICs{true};
                model,
                rng,
                dataloader,
                loss_fn = MSELoss(),
                ad_backend = Lux.AutoZygote(),
                verbose_frequency = 10,
                opt, 
                n_epochs, 
                kwargs...)

    dataloader = tokenize(dataloader)

    ic_list = ParameterLayer[]
    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        push!(ic_list, ParameterLayer(constraint = NoConstraint(), init_value = (;u0)))
    end
    ics = InitialConditions(ic_list)

    function feature_wrapper((token, tsteps_batch))
        return [
            (;u0 = token[i],
            saveat = tsteps_batch[:, i], 
            tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
            )
            for i in 1:batchsize
        ]
    end

    ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(feature_wrapper), initial_conditions = ics, model = model)

    ps, st = Lux.setup(rng, ode_model_with_ics)
    train_state = Training.TrainState(ode_model_with_ics, ps, st, opt)


    for epoch in 1:n_epochs
        tot_loss = 0.
        for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader
            _, loss, _, train_state = Training.single_train_step!(
                ad_backend, 
                loss_fn, 
                ((batched_tokens, batched_tsteps), batched_segments),
                train_state)
            tot_loss += loss
        end
        if epoch % verbose_frequency == 0
            println("Epoch $epoch: Total Loss = ", tot_loss)
        end
    end
    return train_state
end

get_parameter_values(train_state::Training.TrainState) = train_state.parameters.model