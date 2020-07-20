module FluxModels

using Reexport
include("./VisionModel/VisionModel.jl")
@reexport using .VisionModel

include("VisionModel/alexnet.jl")
include("VisionModel/resnet.jl")

end
