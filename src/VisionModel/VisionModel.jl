module VisionModel

using Flux

# Models
include("alexnet.jl")
include("convnb.jl")
include("vgg.jl")
include("resnet.jl")
include("googlenet.jl")

end
