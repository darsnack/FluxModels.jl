using Flux
using FluxModels
using Test

@testset "Vision Models" begin
    include("VisionModel/alexnet.jl")
    include("VisionModel/resnet.jl")
    include("VisionModel/vgg.jl")
    include("VisionModel/googlenet.jl")
end
