using FluxModels:alexnet
using Test

@testset "alexnet" begin
  m = alexnet()
  @test size(m(rand(256, 256, 3, 50))) == (1000, 50)
end

