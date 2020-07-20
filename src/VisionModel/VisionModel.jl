module VisionModel

using Flux, Images

# Models
include("alexnet.jl")
include("vgg.jl")
include("resnet.jl")

end
