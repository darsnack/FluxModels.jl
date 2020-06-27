using Flux
using FluxModels
using Test

@testset "FluxModels" begin
    include("FluxModels.jl")
end

@testset "alexnet" begin
    include("alexnet.jl")
end

@testset "resnet" begin
    include("resnet.jl")
end
