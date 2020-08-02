@testset "AlexNet" begin
  model = alexnet()
  @test size(model(rand(Float32, 256, 256, 3, 50))) == (1000, 50)
end
