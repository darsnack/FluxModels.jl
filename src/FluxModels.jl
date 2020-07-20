module FluxModels

using Reexport
include("./VisionModel/VisionModel.jl")
@reexport using .VisionModel

end
