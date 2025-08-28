abstract type AbstractEcosystemModel end


function (model::AbstractEcosystemModel)(components, u, ps, t)
    p = components.parameters(ps.parameters)
    ũ = max.(u, zero(eltype(u)))
    du = ũ .* (intinsic_growth_rate(model, p, t) .- competition(model, ũ, p) .+ feed_pred_gains(model, ũ, p))
    return du
end

intinsic_growth_rate(::AbstractEcosystemModel, p, t) = p.r

function feeding(m::AbstractEcosystemModel, u, p)
    Warr, Harr, qarr = create_sparse_matrices(m, p)

    return qarr .* Warr ./ (one(eltype(u)) .+ qarr .* Harr .* (Warr * u))
end

function feed_pred_gains(model::AbstractEcosystemModel, u, p)
    F = feeding(model, u, p)
    return  (F .- F') * u
end