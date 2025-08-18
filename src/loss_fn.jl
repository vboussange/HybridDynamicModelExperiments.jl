## Loss function
struct LogMSELoss end #TODO: this should be included in PiecewiseInference.jl
function (::LogMSELoss)(data, pred)
    T = eltype(data)
    data = max.(data, T(1f-6)) # we do not tolerate negative data
    pred = max.(pred, T(1f-6)) #
    if size(data) != size(pred) # preventing Zygote to crash
        return Inf
    end

    l = sum((log.(data) .- log.(pred)).^2) / length(data)
    return l
    # if l isa Number # preventing any other reason for Zygote to crash
    #     return l
    # else 
    #     return Inf
    # end
end


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