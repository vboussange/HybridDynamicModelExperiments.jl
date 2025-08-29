## Loss function
# See
# https://github.com/LuxDL/Lux.jl/blob/13045f78bb98c57081494a1fb8ed8e6dbf151bb8/src/helpers/losses.jl#L763
function _log_mseloss(data, pred; epsilon=1e-6)
    T = eltype(data)
    data = max.(data, T(epsilon)) # we do not tolerate negative data
    pred = max.(pred, T(epsilon)) #
    if size(data) != size(pred) # preventing Zygote to crash
        return T(Inf)
    end

    return (log.(data) .- log.(pred)).^2
    # if l isa Number # preventing any other reason for Zygote to crash
    #     return l
    # else 
    #     return Inf
    # end
end

function LogMSELoss(;agg=mean)
    return GenericLossFunction(_log_mseloss; agg)
end


# struct LossLikelihoodPartialObs end
# function (::LossLikelihoodPartialObs)(data, pred, rng)
#     # this happens for initial conditions
#     size(data, 1) == 3 ? data = data[2:3,:] : nothing
#     # we discard prediction of the resource abundance
#     pred = pred[2:end,:]
#     if any(pred .<= 0.) # we do not tolerate non-positive ICs -
#         return Inf
#     elseif size(data,2) != size(pred, 2) # preventing Zygote to crash
#         println("hey")
#         @show size(data)
#         @show size(pred)
#         return Inf
#     end

#     l = sum((log.(data) .- log.(pred)).^2)

#     if l isa Number # preventing any other reason for Zygote to crash
#         return l
#     else 
#         return Inf
#     end
# end