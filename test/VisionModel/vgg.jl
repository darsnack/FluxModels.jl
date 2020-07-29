sing Flux
using FluxModels
using Test

@testset "ResNet" begin
  for model in [vgg11, vgg11bn, vgg13, vgg13bn, vgg15, vgg15bn, vgg17, vgg17bn, vgg19, vgg19bn]
    m = model()
    @test size(m(rand(Float32, 256, 256, 3, 50))) == (1000, 50)
  end
end

