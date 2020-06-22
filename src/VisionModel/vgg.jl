const configs = Dict(:A => [(64,1) (128,1) (256,2) (512,2) (512,2)],
                     :B => [(64,2) (128,2) (256,2) (512,2) (512,2)],
                     :D => [(64,2) (128,2) (256,3) (512,3) (512,3)],
                     :E => [(64,2) (128,2) (256,4) (512,4) (512,4)])

# Build a VGG block
#  ifilters: number of input filters
#  ofilters: number of output filters
#  batchnorm: add batchnorm (see below for the problem of biases in Conv)
function vgg_block(ifilters, ofilters, depth, batchnorm)
  k = (3,3)
  p = (1,1)
  i = Flux.glorot_uniform
  layers = []
  for l in 1:depth
    if batchnorm
      # Conv with BatchNorm must have no biases, not possible with Flux v0.10.4, however it seems
      # available in Flux#master
      push!(layers, Conv(k, ifilters=>ofilters, pad=p, init=i))
      push!(layers, BatchNorm(ofilters, relu))
    else
      push!(layers, Conv(k, ifilters=>ofilters, relu, pad=p, init=i))
    end
    ifilters = ofilters
  end
  return layers
end
