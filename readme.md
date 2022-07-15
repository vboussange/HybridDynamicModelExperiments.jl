This repository contains the code used for the paper 

> *Mini-batching ecological data to improve ecosystem models with Machine Learning*, Boussange, V., Vilimelis-Aceituno, P., Pellissier, L., (2022)

- `code/` contains all scripts related to the simulation runs
- `figure/` contains all scripts to generate the manuscript figures and crunch the raw simulation results.

All scripts are written in the Julia programming language. A short description of the purpose of each script is placed in each script preamble.
The scripts can be executed out of the box by activating the environment stored in the `Project.toml` and `Manifest.toml` files in the root folder.

To activate the environment in an interactive session, type in the Julia REPL

```julia
julia>] activate .
julia>] instantiate
```
To simply run a script, type in the terminal
```
> julia --project=. name_of_the_script.jl
```