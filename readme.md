This repository contains the code used for the paper 

> *Partitioning time series to improve process-based models with machine learning*, Boussange, V., Vilimelis-Aceituno, P., Schäfer, F., Pellissier, L., (2024)

## Content
- `src/` contains utility functions and types specific to the experiments presented in the manuscript, such as e.g. models and the loss function
- `scripts/` contains all scripts related to the actual simulation runs
- `figure/` contains all scripts to generate the manuscript figures and crunch the raw simulation results.


## Installation
All scripts are written in the Julia programming language. We recommend installing Julia with [`juliaup`](https://github.com/JuliaLang/juliaup).
The scripts can be executed out of the box by activating the environment stored in the `Project.toml`, `CondaPkg.toml` and `Manifest.toml` files in the root folder, but before doing so you are required to add a registry that tracks `PiecewiseInference.jl` and a dependency called `ParametricModels.jl`. To do so, open Julia and type the following

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