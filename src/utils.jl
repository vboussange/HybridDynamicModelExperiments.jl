function save_results(pathfile; results, kwargs...)
    println("saving...")
    dir = joinpath(dirname(pathfile), "results", string(today()))
    !isdir(dir) && mkpath(dir)
    namefile = split(split(pathfile, "/")[end], ".")[1]
    jldsave(joinpath(dir, namefile*".jld2"); results, kwargs...)

    col_to_text = eltype.(results[1,:] |> Vector) .<: Union{Real,String}
    open(joinpath(dir, namefile)*".txt", "w") do file
        println(file, results[:,col_to_text])
    end
    println("saved in $dir")
end

using PiecewiseInference
import PiecewiseInference: AbstractODEModel
function validate(infres::InferenceResult, ode_data, true_model::AbstractODEModel; length_horizon = nothing)
    loss_likelihood = infres.infprob.loss_likelihood
    tsteps = true_model.mp.kwargs[:saveat]
    mystep = tsteps[2]-tsteps[1]
    ranges = infres.ranges
    isnothing(length_horizon) && (length_horizon = length(ranges[1]))
    tsteps_forecast = range(start = tsteps[end]+mystep, step = mystep, length=length_horizon)

    forcasted_data = forecast(infres, tsteps_forecast) |> Array
    true_forecasted_data = simulate(true_model; 
                                    u0 = ode_data[:,ranges[end][1]],
                                    tspan = (tsteps[ranges[end][1]], tsteps_forecast[end]), 
                                    saveat = tsteps_forecast) |> Array

    loss_likelihood(forcasted_data, true_forecasted_data, nothing)
end

using ForwardDiff
function params_sensibility(ode_data, true_model::AbstractODEModel, loss_likelihood)
    lossfn(p) =  loss_likelihood(simulate(true_model; p), ode_data, nothing)
    diag(ForwardDiff.hessian(lossfn, true_model.mp.p))
end