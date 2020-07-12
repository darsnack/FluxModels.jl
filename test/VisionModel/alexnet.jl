using Flux
using Test

@testset "Testing AlexNet Model" begin
  model = alexnet()
  @test size(model(rand(256, 256, 3, 50))) == (1000, 50)
end
