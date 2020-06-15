module VisionModel

using Flux, Images, REPL

# Models
export AlexNet

include("AlexNet.jl")

end module
