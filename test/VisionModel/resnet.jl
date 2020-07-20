using Flux
using FluxModels
using Test

@testset "ResNet" begin
  for MODEL in [
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ]   
    model = MODEL()
    @test size(model(rand(256, 256, 3, 50))) == (1000, 50)
  end
end
