using Dates
using DataFrames
using JLD2
import LibGit2
using Random

function save_results(pathfile; results::DataFrame, kwargs...)
    println("saving...")
    repo = LibGit2.GitRepoExt(".")
    head_commit = LibGit2.head(repo)
    hash = string(LibGit2.GitHash(head_commit))[1:7] # short hash
    dir = joinpath(dirname(pathfile), "results")
    !isdir(dir) && mkpath(dir)
    namefile = split(split(pathfile, "/")[end], ".")[1] * "_" * hash
    jldsave(joinpath(dir, namefile*".jld2"); results, kwargs...)

    col_to_text = eltype.(results[1,:] |> Vector) .<: Union{Real,String}
    open(joinpath(dir, namefile)*".txt", "w") do file
        println(file, results[:,col_to_text])
    end
    println("saved in $(joinpath(dir, namefile))")
end

# Generate noisy data
function generate_noisy_data(data, noise, rng = Random.default_rng())
    return data .* exp.(noise * randn(rng, size(data)))
end

function pop(nt::NamedTuple, key)
    value = getproperty(nt, key)
    nt = Base.structdiff(nt, NamedTuple{(key,)})
    return (value, nt)
end