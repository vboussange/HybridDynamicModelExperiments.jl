
struct VIBackend <: AbstractOptimBackend end

function train(::VIBackend,
                model::AbstractLuxLayer,
                dataloader::SegmentedTimeSeries,
                experimental_setup::InferICs;
                rng=Random.default_rng(),
                q_init = q_meanfield_gaussian,
                n_iterations, 
                kwargs...)

    dataloader = tokenize(dataloader)

    xs = []
    ys = []
    ic_list = ParameterLayer[]
    u0_priors = []

    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        t0 = segment_tsteps[1]
        push!(xs, (;u0 = tok, saveat = segment_tsteps, tspan = (segment_tsteps[1], segment_tsteps[end])))
        push!(ys, segment_data)
        if isa(experimental_setup, InferICs{true})
            push!(ic_list, ParameterLayer(init_state_value = (;t0)))
            push!(u0_priors, (;u0 = arraydist(datadistrib.(u0))))
        elseif isa(experimental_setup, InferICs{false})
            push!(ic_list, ParameterLayer(init_state_value = (;t0, u0)))
            push!(u0_priors, (;))
        end
    end
    ics = InitialConditions(ic_list)
    u0_priors = NamedTuple{ntuple(i -> Symbol(:u0_, i), length(ic_list))}(u0_priors)

    ode_model_with_ics = Chain(initial_conditions = ics, model = model)
    priors = (initial_conditions = u0_priors, model = model_priors)

    ps_init, st = Lux.setup(rng, ode_model_with_ics)
    st_model = StatefulLuxLayer{true}(ode_model_with_ics, ps_init, st)

    turing_fit = create_turing_model(priors, datadistrib, st_model)
    turing_model = turing_fit(xs, ys)
    q_avg, q_last, info, state = vi(rng, turing_model, q_init(rng, turing_model), n_iterations; kwargs...)
    # best_ps = get_best_parameters(chains, ps_init)
    # best_model = StatefulLuxLayer{true}(model, best_ps, st)
    return (;q_avg, q_last, info, state)
end

# function train(::MCSamplingBackend, 
#                 ::InferICs{false};
#                 model,
#                 rng=Random.default_rng(),
#                 dataloader,
#                 datadistrib,
#                 model_priors,
#                 sampler = NUTS(; adtype=AutoForwardDiff()),
#                 n_iterations,
#                 kwargs...)

#     dataloader = tokenize(dataloader)

#     xs = []
#     ys = []
#     for tok in tokens(dataloader)
#         segment_data, segment_tsteps = dataloader[tok]
#         push!(xs, (;u0 = segment_data[:, 1], saveat = segment_tsteps, tspan = (segment_tsteps[1], segment_tsteps[end])))
#         push!(ys, segment_data)
#     end

#     ps_init, st = Lux.setup(rng, model) 
#     st_model = StatefulLuxLayer{true}(model, ps_init, st)
#     turing_fit = create_turing_model(model_priors, datadistrib, st_model)

#     chains = sample(rng, turing_fit(xs, ys), sampler, n_iterations, kwargs...)
#     best_ps = get_best_parameters(chains, ps_init)
#     best_model = StatefulLuxLayer{true}(model, best_ps, st)
#     return (;best_model, chains)
# end

# TODO: this is not the right approach; you rather want to sample from the posterior distrib.
# function get_best_parameters(chains::Chains, ps)
#     lp = chains[:lp]
#     max_idx = argmax(lp)
#     ps_vec = reshape(chains[max_idx[1], collect(values(chains.info.varname_to_symbol)), max_idx[2]] |> Array, :)
#     return _vector_to_parameters(ps_vec, ps)
# end
