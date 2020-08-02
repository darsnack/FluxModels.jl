module VisionModel

using Flux

# Models
include("alexnet.jl")
include("convnb.jl")
include("vgg.jl")
include("resnet.jl")
include("googlenet.jl")

export  alexnet,
        vgg11, vgg11bn, vgg13, vgg13bn, vgg16, vgg16bn, vgg19, vgg19bn,
        ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
        googlenet

end
