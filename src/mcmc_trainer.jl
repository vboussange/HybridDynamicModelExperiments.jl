import Turing: @model, NUTS, sample, Chains, arraydist, q_meanfield_gaussian, vi
import DynamicPPL
import DynamicPPL: @varname, VarName
using Distributions
import Lux
import Lux:fmap
import Functors: @leaf, fmap_with_path
using ComponentArrays
using ConcreteStructs: @concrete

@concrete struct MCMCBackend <: AbstractOptimBackend
    sampler
    n_iterations::Int
    datadistrib
    kwargs
end

function MCMCBackend(sampler = NUTS(; adtype=AutoForwardDiff()),
                    n_iterations,
                    datadistrib = Normal(),
                    ;kwargs...)
    return MCMCBackend(sampler, n_iterations, datadistrib, kwargs)
end


function _vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

# # TODO: this is to be revised, type inference fails here
# function _parameters_to_vector(ps::NamedTuple)
#     total = Lux.parameterlength(ps)
#     out = Vector(undef, total)   # parameters are expected to be numeric (Float64)
#     i = 1
#     function put(x)
#         len = length(x)
#         out[i:(i + len - 1)] .= vec(x)
#         i += len
#         return nothing
#     end
#     Lux.fmap(put, ps)
#     @assert i - 1 == total
#     return out
# end

# required for handling prior distributions in NamedTuples
Lux.parameterlength(dist::Distributions.Distribution) = length(dist)
Base.vec(dist::Product) = dist.v
@leaf Distributions.Distribution


function create_turing_model(ps_priors, data_distrib, st_model)
    function generated_model(model, varinfo, xs, ys)
        # Use a Ref to allow updating varinfo inside the fmap_with_path closure
        varinfo_ref = Ref(varinfo)
        
        # Function to handle each node in the param_prior structure
        function handle_node(path, node::Distributions.Distribution)
            # Generate variable name from path
            varname = Symbol(join(path, "_"))
            # Sample parameter and update varinfo
            value, new_varinfo = DynamicPPL.tilde_assume!!(model.context, node, VarName{varname}(), varinfo_ref[])
            varinfo_ref[] = new_varinfo
            return value
        end

        handle_node(path, node) = (;)
        
        # Apply fmap_with_path to sample all parameters and maintain structure
        ps = fmap_with_path(handle_node, ps_priors) |> ComponentArray
        
        # Update varinfo after sampling all parameters
        varinfo = varinfo_ref[]

        # Observe data points
        for i in eachindex(xs)
            preds = st_model(xs[i], ps)
            dists = data_distrib.(preds)
            _retval, varinfo = DynamicPPL.tilde_observe!!(
                model.context, arraydist(dists), ys[i], @varname(ys[i]), varinfo
            )
        end
        
        return nothing, varinfo
    end
    
    return (xs, ys) -> DynamicPPL.Model(generated_model, (; xs, ys))
end

function train(backend::MCMCBackend, 
                experimental_setup::InferICs;
                model::AbstractLuxLayer,
                rng=Random.default_rng(),
                dataloader)

    dataloader = tokenize(dataloader)

    # TODO: 

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
            push!(ic_list, BayesianLayer(ParameterLayer(init_value = (;u0), init_state_value = (;t0)), 
                                        (;u0 = arraydist(datadistrib.(u0)))))
        elseif isa(experimental_setup, InferICs{false})
            push!(ic_list, BayesianLayer(ParameterLayer(init_state_value = (;t0, u0)),
                                        (;)))
        end
    end
    ics = InitialConditions(ic_list)

    ode_model_with_ics = Chain(initial_conditions = ics, model = model)
    priors = getpriors(ode_model_with_ics)

    ps_init, st = Lux.setup(rng, ode_model_with_ics)
    st_model = StatefulLuxLayer{true}(ode_model_with_ics, ps_init, st)

    turing_fit = create_turing_model(priors, backend.datadistrib, st_model)

    chains = sample(rng, turing_fit(xs, ys), backend.sampler, backend.n_iterations; backend.kwargs...)
    # best_ps = get_best_parameters(chains, ps_init)
    # best_model = StatefulLuxLayer{true}(model, best_ps, st)
    return (;chains, st_model)
end

function train(::VIBackend, 
                experimental_setup::InferICs;
                model::AbstractLuxLayer,
                rng=Random.default_rng(),
                dataloader,
                datadistrib,
                model_priors,
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

# function train(::MCMCBackend, 
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
