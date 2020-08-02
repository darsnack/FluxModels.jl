using Flux
using FluxModels
using Test

@testset "VGG" for model in [vgg11, vgg11bn, vgg13, vgg13bn, vgg16, vgg16bn, vgg19, vgg19bn]
  imsize = rand(Float32, 256, 256, 3, 50)
  m = model(imsize)  
  
  @test size(m) == (1000, 50)
end

