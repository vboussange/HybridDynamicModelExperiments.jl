#=
Short exampling showcasing the fit of a 3 species model.
=#
cd(@__DIR__)

using Dates, DataFrames, XLSX, CSV
DATA_PATH = "beninca_data.xlsx"



# importing data
data = XLSX.readtable(DATA_PATH, "Interpolated Data") |> DataFrame
data[!, 1] = Dates.Date.(data[!, 1]) .- Year(1900) 
data[!, 2:end] = Float32.(data[!, 2:end]) ./ 100f0 # converting to float and normalizing

forcing = XLSX.readtable(DATA_PATH, "Temperature Data Fig.2A") |> DataFrame
forcing2 = XLSX.readtable(DATA_PATH, "Wave and Wind Data Fig. S7") |> DataFrame
forcing_merged = innerjoin(forcing, forcing2, on = :Date, makeunique=true)
forcing_merged[!, 1] = Date.(forcing_merged[!, 1])

# Filling missing values in forcing data
for row in 2:size(forcing_merged, 1)
    for col in 2:size(forcing_merged, 2)
        if ismissing(forcing_merged[row, col])
            forcing_merged[row, col] = forcing_merged[row - 1, col]
        end
    end
end
forcing_merged[!, 2:end] = Float32.(forcing_merged[!, 2:end])

CSV.write(joinpath(@__DIR__, "beninca_data.csv"), data)
CSV.write(joinpath(@__DIR__, "forcing_merged.csv"), forcing_merged)