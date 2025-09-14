using HybridModelling
using Plots

datasize = (1, 100)
tsteps = collect(1.0:1.0:datasize[2])
data = reshape(sin.(0.2e0 * tsteps), 1, :)

segmentsize = 4
valid_length = 1
dataloader_train = tokenize(SegmentedTimeSeries((tsteps, data); segmentsize, shift = segmentsize + valid_length, partial_segment=false))
dataloader_valid = tokenize(SegmentedTimeSeries((tsteps[segmentsize+1:end], data[:, segmentsize+1:end]); segmentsize = valid_length, shift = segmentsize+valid_length, partial_segment=false))

println("Number of training segments: ", length(tokens(dataloader_train)))
println("Number of validation segments: ", length(tokens(dataloader_valid)))

ax = Plots.plot()
for tok in tokens(dataloader_train)
    segment_tsteps, segment_data = dataloader_train[tok]
    Plots.scatter!(ax, segment_tsteps, segment_data', label = "", c = :blue, alpha=0.3)
end
display(ax)

for tok in tokens(dataloader_valid)
    segment_tsteps, segment_data = dataloader_valid[tok]
    Plots.scatter!(ax, segment_tsteps, segment_data', label = "", linestyle = :dash, c=:red, alpha=0.3)
end
display(ax)
