sing Flux
using FluxModels
using Test

@testset "ResNet" begin
    m = googlenet()
    @test size(m(rand(Float32, 224, 224, 3, 50))) == (1000, 50)
  end
end


