using ComponentArrays
using Optimisers
using ADTypes
using ConcreteStructs: @concrete
import HybridDynamicModels: SegmentedTimeSeries

function HybridDynamicModels.train(backend::SGDBackend,
        model::AbstractLuxLayer,
        dataloader_train::SegmentedTimeSeries,
        experimental_setup::WithValidation,
        rng = Random.default_rng(),
        luxtype = Lux.f64)
    dataloader_train = luxtype(tokenize(dataloader_train))
    dataloader_valid = luxtype(tokenize(experimental_setup.dataloader))

    @assert length(tokens(dataloader_train))==length(tokens(dataloader_valid)) "The training and validation dataloaders must have the same number of segments"

    ics_layer = _get_ic_layer(dataloader_train, experimental_setup)

    if !is_ics_estimated(experimental_setup)
        ics_layer = Lux.Experimental.FrozenLayer(ics_layer)
    end

    ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(_feature_wrapper),
        initial_conditions = ics_layer, model = model)

    ps, st = luxtype(Lux.setup(rng, ode_model_with_ics))
    ps = ComponentArray(ps) # We transforms ps to support all sensealg from SciMLSensitivity

    # Initialize training state
    train_state = Training.TrainState(ode_model_with_ics, ps, st, backend.opt)
    best_ps = ps.model
    best_st = st.model
    segment_ics = _get_ic_values(dataloader_train, ics_layer, ps.initial_conditions, st.initial_conditions)
    best_loss = luxtype(Inf)
    for epoch in 1:(backend.n_epochs)
        train_loss = luxtype(0.0)
        for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader_train
            _, loss, _, train_state = Training.single_train_step!(
                backend.adtype,
                backend.loss_fn,
                ((batched_tokens, batched_tsteps), batched_segments),
                train_state)
            train_loss += loss
        end

        valid_loss = 0.0
        ps, st = get_parameter_values(train_state), get_state_values(train_state)
        segment_ics = _get_ic_values(dataloader_train, ics_layer, ps.initial_conditions, st.initial_conditions)
        for tok in tokens(dataloader_valid)
            u0 = segment_ics[tok].u0
            _, segment_tsteps_train = dataloader_train[tok]
            t0 = segment_tsteps_train[1]
            data_valid, saveat = dataloader_valid[tok]
            tspan = (t0, saveat[end])
            data_pred = model((; u0, saveat, tspan), ps.model, st.model)[1][
                :, :, 1
            ]
            valid_loss += backend.loss_fn(data_pred, data_valid)
        end
        @debug "Train loss: $train_loss"
        @debug "Validation loss: $valid_loss"
        if valid_loss < best_loss
            best_ps = ps.model
            best_st = st.model
            best_loss = train_loss
        end
        backend.callback(train_loss, epoch, train_state)
    end

    return (; ps = best_ps, st = best_st, ics = segment_ics)
end
