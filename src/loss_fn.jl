function loss_fn(data, params, pred, rg, ic_term, prior_scaling; scaling_var = 1.)
    if any(pred .<= 0.) # we do not tolerate non-positive ICs -
        return Inf
    elseif size(data) != size(pred) # preventing Zygote to crash
        return Inf
    end

    l = mean(mean((data .- pred).^2, dims=2) ./ scaling_var.^2)
    l += mean((data[:,1] - pred[:,1]).^2 ./ scaling_var.^2 ) * ic_term # putting more weights on initial conditions
    # l += 10000 * sum(pred .< 1e-2) # preventing too low values
    if l isa Number # preventing any other reason for Zygote to crash
        return l
    else 
        return Inf
    end
end

function loss_fn_log_prior_param(;data, params, pred, rg, ic_term, var_noise, prior_param, scale_prior_param)

    if any(pred .<= 0.) # we do not tolerate non-positive ICs -
        return Inf
    elseif size(data) != size(pred) # preventing Zygote to crash
        return Inf
    end

    # observations
    l = mean((log.(data) .- log.(pred)).^2 ./ var_noise) # standard loss
    # parameter prior
    for k in keys(prior_param)
        l += sum(((reshape(params[k],:) .- mean(prior_param[k])).^2 ./ var(prior_param[k]))) * scale_prior_param
    end
    # initial conditions
    l += mean((log.(data[:,1]) - log.(pred[:,1])).^2 ./ var_noise) * ic_term # adding more importance on initial conditions

    if l isa Number # preventing any other reason for Zygote to crash
        return l
    else 
        return Inf
    end
end

# using Distributions to facilitate loglikelihood calculations
# Noise should be lognormal, noise_distrib should be of type MvNormal with zero mean
function loss_fn_lognormal_distrib(data, pred, noise_distrib)
    if any(pred .<= 0.) # we do not tolerate non-positive ICs -
        return Inf
    elseif size(data) != size(pred) # preventing Zygote to crash
        return Inf
    end

    l = 0.

    # observations
    ϵ = log.(data) .- log.(pred)
    for i in 1:size(ϵ, 2)
        l += logpdf(noise_distrib, ϵ[:, i])
    end
    # l /= size(ϵ, 2) # TODO: this bit is unclear

    if l isa Number # preventing any other reason for Zygote to crash
        return - l
    else 
        return Inf
    end
end

## Loss function
struct LossLikelihood end
function (::LossLikelihood)(data, pred, rng)
    if any(pred .<= 0.) # we do not tolerate non-positive ICs -
        return Inf
    elseif size(data) != size(pred) # preventing Zygote to crash
        return Inf
    end

    l = sum((log.(data) .- log.(pred)).^2)

    if l isa Number # preventing any other reason for Zygote to crash
        return l
    else 
        return Inf
    end
end

(l::LossLikelihood)(data, pred) = l(data, pred, nothing) # method required for loss_u0_prior in InferenceProblem

struct LossLikelihoodPartialObs end
function (::LossLikelihoodPartialObs)(data, pred, rng)
    # this happens for initial conditions
    size(data, 1) == 3 ? data = data[2:3,:] : nothing
    # we discard prediction of the resource abundance
    pred = pred[2:end,:]
    if any(pred .<= 0.) # we do not tolerate non-positive ICs -
        return Inf
    elseif size(data,2) != size(pred, 2) # preventing Zygote to crash
        println("hey")
        @show size(data)
        @show size(pred)
        return Inf
    end

    l = sum((log.(data) .- log.(pred)).^2)

    if l isa Number # preventing any other reason for Zygote to crash
        return l
    else 
        return Inf
    end
end

(l::LossLikelihoodPartialObs)(data, pred) = l(data, pred, nothing) # method required for loss_u0_prior in InferenceProblem
