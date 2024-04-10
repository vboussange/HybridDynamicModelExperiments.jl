This repository contains the code used for the paper 

> *Partitioning time series to improve process-based models with machine learning*, Boussange, V., Vilimelis-Aceituno, P., Schäfer, F., Pellissier, L., (2024)

## Content
- `src/` contains utility functions and types specific to the experiments presented in the manuscript, such as e.g. models and the loss function
- `scripts/` contains all scripts related to the actual simulation runs
- `figure/` contains all scripts to generate the manuscript figures and crunch the raw simulation results.


## Installation
All scripts are written in the Julia programming language. We recommend installing Julia with [`juliaup`](https://github.com/JuliaLang/juliaup).
The scripts can be executed by activating the environment stored in the `Project.toml`, `CondaPkg.toml` and `Manifest.toml` files in the root folder. However, this environment depends on `PiecewiseInference.jl` and `ParametricModels.jl`, which are packages not yet registered in the official Julia repository. To circumvent this issue, you simply need to add a custom registry tracking those unregistered packages to your Julia installation. Type in the Julia REPL

```julia
julia> using Pkg
julia>] registry add https://github.com/vboussange/VBoussangeRegistry.git
```
The custom registry is now registered on your machine, and you can activate the environment of the repo.
To activate the environment in an interactive session, type in the Julia REPL

```julia
julia>] activate .
julia>] instantiate
```


## Getting started
To run a script, type in the terminal
```
julia --project=. name_of_the_script.jl
```

Start with
```
julia --project=. scripts/illustrate_converence/illustrate_convergence.jl
```