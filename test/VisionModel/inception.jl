@testset "InceptionV3" begin
    m = inception3()
    @test size(m(rand(Float32, 299, 299, 3, 50))) == (1000, 50)
end
