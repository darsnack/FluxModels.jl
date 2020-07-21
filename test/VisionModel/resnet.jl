using Flux
using FluxModels
using Test

@testset "ResNet" begin
    @test size(ResNet18(rand(256, 256, 3, 50))) == (1000, 50),
    @test size(ResNet34(rand(256, 256, 3, 50))) == (1000, 50),
    @test size(ResNet50(rand(256, 256, 3, 50))) == (1000, 50),
    @test size(ResNet101(rand(256, 256, 3, 50))) == (1000, 50),
    @test size(ResNet152(rand(256, 256, 3, 50))) == (1000, 50)
end
