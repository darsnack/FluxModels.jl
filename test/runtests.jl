using Flux
using FluxModels
using Test

@info "Testing Flux Models..."
@testset "Testing Vision Models" begin
    include("VisionModel/alexnet.jl")
    include("VisionModel/resnet.jl")
end
