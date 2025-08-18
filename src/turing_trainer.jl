function train(::TuringBackend, 
                ::InferICs{true};
                model,
                rng,
                dataloader,
                sampler = HMC(0.05, 4; adtype=AutoForwardDiff()),
                n_iterations)

    dataloader = tokenize(dataloader)

    xs = []
    ys = []
    ic_list = ParameterLayer[]

    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        push!(xs, (;u0 = tok, saveat = segment_tsteps, tspan = (segment_tsteps[1], segment_tsteps[end])))
        push!(ys, segment_data)
        push!(ic_list, ParameterLayer(constraint = NoConstraint(), init_value = (;u0)))
    end
    ics = InitialConditions(ic_list)

    ode_model_with_ics = Chain(initial_conditions = ics, model = model)

    ps_init, st = Lux.setup(rng, ode_model_with_ics) 
    st_model = StatefulLuxLayer{true}(ode_model_with_ics, ps_init, st)

    @model function turing_fit(xs, ys, ps)

        # Order for defining parameters matters !
        # when converting back to named tuples
        # should be declared in the same order as ps
        u0s = Vector{Vector}(undef, length(xs))
        for i in eachindex(xs)
            _u0 = ys[i][:, 1]
            u0s[i] ~ arraydist(LogNormal.(log.(_u0), σ))
        end

        b ~ Product(fill(Uniform(0.1, 1e0),2))

        parameters = vcat(u0s..., b)
        ps_tur = vector_to_parameters(parameters, ps)

        for i in eachindex(xs)
            preds = st_model(xs[i], ps_tur)
            ys[i] ~ arraydist(LogNormal.(log.(preds), σ))
        end
        return nothing
    end

    ch = sample(bayes_fit(xs, ys, ps_init), sampler, n_iterations)
    return ch
end

function train(::TuringBackend, 
                ::InferICs{false};
                model,
                rng,
                sampler = HMC(0.05, 4; adtype=AutoForwardDiff()),
                n_iterations,
                kwargs...)

    dataloader = tokenize(dataloader)

    xs = []
    ys = []
    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        push!(xs, (;u0 = segment_data[:, 1], saveat = segment_tsteps, tspan = (segment_tsteps[1], segment_tsteps[end])))
        push!(ys, segment_data)
    end

    ps_init, st = Lux.setup(rng, model) 
    st_model = StatefulLuxLayer{true}(model, ps_init, st)

    @model function turing_fit(xs, ys)
        b ~ Product(fill(Uniform(0.1, 1e0),2))
        ps = ComponentArray(;params = (;b))
        for i in eachindex(xs)
            preds = st_model(xs[i], ps)
            ys[i] ~ arraydist(LogNormal.(log.(preds), σ))
        end
        return nothing
    end

    ch = sample(turing_fit(xs, ys), sampler, n_iterations)
    return ch
end

get_parameter_values(ch::AbstractMCMC) = nothing # TODO: to complete
get_loss(ch::AbstractMCMC) = nothing