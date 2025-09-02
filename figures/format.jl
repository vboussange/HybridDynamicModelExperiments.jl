#=
Settings for pyplot
=#
using PythonCall
# using PyCall

const matplotlib = pyimport("matplotlib")
const plt = pyimport("matplotlib.pyplot")
const Line2D = matplotlib.lines.Line2D

rcParams = plt."rcParams"

rcParams["font.size"] = 9
rcParams["axes.titlesize"] = 10
rcParams["axes.labelsize"] = 10
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["legend.fontsize"] = 8
rcParams["figure.titlesize"] = 10
rcParams["lines.markersize"] = 3

color_palette = ["tab:blue", "tab:red", "tab:green"]
