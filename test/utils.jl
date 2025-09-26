using Test
import HybridDynamicModelExperiments: generate_noisy_data, pop, save_results
using DataFrames
@testset "pop" begin
    nt = (a=1, b="x", c=[1,2])
    val, rest = pop(nt, :a)
    @test val == 1
    @test hasproperty(rest, :b) && getproperty(rest, :b) == "x"
    @test !hasproperty(rest, :a)
end



@testset "save_results" begin
    td = mktempdir(@__DIR__)
    cd(td) do
        # initialize a git repo and make an initial commit so save_results can read HEAD
        df = DataFrame(x = [1,2], y = ["a", "b"])
        save_results("myfile.jl"; results = df)

        results_dir = joinpath(".", "results")
        @test isdir(results_dir)
        files = readdir(results_dir)
        @test any(endswith.(files, ".jld2"))
        @test any(endswith.(files, ".txt"))
    end
end
