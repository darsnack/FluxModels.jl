using Flux
using Test

@testset "Vision Models" begin
    include("VisionModel/alexnet.jl")
    include("VisionModel/resnet.jl")
end
