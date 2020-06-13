module VisionModel

using Flux, Images, REPL
using Flux: @treelike

# Models
export AlexNet

# Datasets
export ImageNet, CIFAR10

include("AlexNet.jl")

end module
