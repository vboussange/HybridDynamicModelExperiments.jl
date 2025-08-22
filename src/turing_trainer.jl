import Turing: @model, HMC, sample, Chains, arraydist
import DynamicPPL
import DynamicPPL: @varname, VarName
using Distributions
import Lux
import Lux:fmap
import Functors: @leaf, fmap_with_path

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
        function handle_node(path, node)
            # Generate variable name from path
            varname = Symbol(join(path, "_"))
            # Sample parameter and update varinfo
            value, new_varinfo = DynamicPPL.tilde_assume!!(model.context, node, VarName{varname}(), varinfo_ref[])
            varinfo_ref[] = new_varinfo
            return value
        end
        
        # Apply fmap_with_path to sample all parameters and maintain structure
        ps = fmap_with_path(handle_node, ps_priors)
        
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


function train(::TuringBackend, 
                ::InferICs{true};
                model::AbstractLuxLayer,
                rng=Random.default_rng(),
                dataloader,
                datadistrib,
                model_priors, # TODO: ideally, model_priors are embedded within `model`, possibly in its state. Depending on the backend, we transform the priors into NamedTransform or the likes
                sampler = HMC(0.05, 4; adtype=AutoForwardDiff()),
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
        push!(xs, (;u0 = tok, saveat = segment_tsteps, tspan = (segment_tsteps[1], segment_tsteps[end])))
        push!(ys, segment_data)
        push!(ic_list, ParameterLayer())
        push!(u0_priors, (;u0 = arraydist(datadistrib.(u0))))
    end
    ics = InitialConditions(ic_list)
    u0_priors = NamedTuple{ntuple(i -> Symbol(:u0_, i), length(ic_list))}(u0_priors)

    ode_model_with_ics = Chain(initial_conditions = ics, model = model)
    priors = (initial_conditions = u0_priors, model = model_priors)

    ps_init, st = Lux.setup(rng, ode_model_with_ics)
    st_model = StatefulLuxLayer{true}(ode_model_with_ics, ps_init, st)

    turing_fit = create_turing_model(priors, datadistrib, st_model)

    ch = sample(rng, turing_fit(xs, ys), sampler, n_iterations, kwargs...)
    return ch
end

function train(::TuringBackend, 
                ::InferICs{false};
                model,
                rng=Random.default_rng(),
                dataloader,
                datadistrib,
                model_priors,
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
    turing_fit = create_turing_model(model_priors, datadistrib, st_model)

    ch = sample(rng, turing_fit(xs, ys), sampler, n_iterations, kwargs...)
    return ch
end

get_parameter_values(ch::Chains) = nothing # TODO: to complete
get_loss(ch::Chains) = nothing