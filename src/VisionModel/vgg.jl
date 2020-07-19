using Flux

# Build a VGG block
#  ifilters: number of input filters
#  ofilters: number of output filters
#  batchnorm: add batchnorm
function vgg_block(ifilters, ofilters, depth, batchnorm)
  k = (3,3)
  p = (1,1)
  layers = []
  for l in 1:depth
    if batchnorm
      w = convfilter(k, ifilters=>ofilters)
      b = Zeros()
      push!(layers, Conv(weight=w, bias=b, pad=p))
      push!(layers, BatchNorm(ofilters, relu))
    else
      push!(layers, Conv(k, ifilters=>ofilters, relu, pad=p))
    end
    ifilters = ofilters
  end
  return layers
end

# Build convolutionnal layers
#  config: :A (vgg11) :B (vgg13) :D (vgg16) :E (vgg19)
#  inchannels: number of channels in input image (3 for RGB)
function convolutional_layers(config, batchnorm, inchannels)
  layers = []
  ifilters = inchannels
  for c in configs[config]
    push!(layers, vgg_block(ifilters, c..., batchnorm)...)
    push!(layers, MaxPool((2,2)))
    ifilters, _ = c
  end
  return layers
end

# Build classification layers
#  imsize: image size
#  nclasses: number of classes
#  fcsize: size of fully connected layers (usefull for smaller nclasses than ImageNet)
#  dropout: dropout obviously
function classifier_layers(imsize, nclasses, fcsize, dropout)
  layers = []
  push!(layers, flatten)
  push!(layers, Dense(Int(prod(imsize) / 2), fcsize, relu))
  push!(layers, Dropout(dropout))
  push!(layers, Dense(fcsize, fcsize, relu))
  push!(layers, Dropout(dropout))
  push!(layers, Dense(fcsize, nclasses))
  push!(layers, softmax)
  return layers
end

function vgg(imsize; config, batchnorm=false, nclasses)
  conv = convolutional_layers(config, batchnorm, 3)
  class = classifier_layers(imsize, nclasses, 4096, 0.5)
  return Chain(conv..., class...)
end

const configs = Dict(:A => [(64,1) (128,1) (256,2) (512,2) (512,2)],
                     :B => [(64,2) (128,2) (256,2) (512,2) (512,2)],
                     :D => [(64,2) (128,2) (256,3) (512,3) (512,3)],
                     :E => [(64,2) (128,2) (256,4) (512,4) (512,4)])

vgg11(imsize; nclasses) =
  vgg(imsize, config=:A, nclasses=nclasses)

vgg11bn(imsize; nclasses) =
  vgg(imsize, config=:A, batchnorm=true, nclasses=nclasses)

vgg13(imsize; nclasses) =
  vgg(imsize, config=:B, nclasses=nclasses)

vgg13bn(imsize; nclasses) =
  vgg(imsize, config=:B, batchnorm=true, nclasses=nclasses)

vgg16(imsize; nclasses) =
  vgg(imsize, config=:D, nclasses=nclasses)

vgg16bn(imsize; nclasses) =
  vgg(imsize, config=:D, batchnorm=true, nclasses=nclasses)

vgg19(imsize; nclasses) =
  vgg(imsize, config=:E, nclasses=nclasses)

vgg19bn(imsize; nclasses) =
  vgg(imsize, config=:E, batchnorm=true, nclasses=nclasses)
