using Flux

basicblock(inplanes::Int, outplanes::Int, downsample::Bool = false) = downsample ? 
  Chain(Conv((3, 3), inplanes => outplanes, stride = 1, pad = 1),
        BatchNorm(outplanes, λ = relu),
        Conv((3, 3), outplanes => outplanes, stride = 2, pad = 1,
        BatchNorm(outplanes, λ=relu)) : 
  Chain(Conv((3, 3), inplanes => outplanes, stride = 1, pad = 1),
        BatchNorm(outplanes, λ = relu),
        Conv((3, 3), outplanes => outplanes, stride = 1, pad = 1,
        BatchNorm(outplanes, λ=relu))

bottleneck(inplanes::Int, outplanes::Int, downsample::Bool = false) = downsample ?
  Chain(Conv((1, 1), inplanes => inplanes, stride = 1),
        BatchNorm(inplanes, λ = relu),
        Conv((3, 3), inplanes => inplanes, stride = 1, pad = 1),
        BatchNorm(inplanes, λ = relu),
        Conv((1, 1), inplanes => outplanes, stride = 2),
        BatchNorm(outplanes, λ = relu)) :
  Chain(Conv((1, 1), inplanes => inplanes, stride = 1),
        BatchNorm(inplanes, λ = relu),
        Conv((3, 3), inplanes => inplanes, stride = 1, pad = 1),
        BatchNorm(inplanes, λ = relu),
        Conv((1, 1), inplanes => outplanes, stride = 1),
        BatchNorm(outplanes, λ = relu))

projection(inplanes::Int, outplanes::Int, stride::Int) = Chain(Conv((1, 1), inplanes => outplanes, stride=stride),
                                                               BatchNorm((outplanes), λ=relu))

identity(inplanes::Int, outplanes::Int, stride::Int) = +

function resnet(block, shortcut, channel_config, block_config)
  layers = []
  push!(layers, Conv((7, 7), 3=>inplanes, stride=(2, 2), pad=(3, 3)))
  push!(layers, BatchNorm(inplanes, λ=relu))
  push!(layers, MaxPool((3, 3), stride=(2, 2), pad=(1, 1)))
  inplanes = 64
  for nrepeats in block_config
    for i in 1:nrepeats-1
      for expansion in channel_config
        outplanes = inplanes * expansion
        push!(layers, SkipConnection(block(inplanes, outplanes, i == 1),
                                     identity(inplanes, outplanes, stride)))
      end
    end 
    inplanes *= 2
    push!(layers, SkipConnection(block(inplanes, outplanes, i == 1),
                                 projection(inplanes, outplanes, stride)))
  end
  push!(layers, AdaptiveMeanPool(1, 1))
  push!(layers, x -> flatten(x, 1))
  push!(layers, Dense(512 * expansion, 1000))
  Flux.testmode!(layers)
  return layers
end

const resnet_config =
Dict("resnet18" => ([1, 1], [2, 2, 2, 2]),
     "resnet34" => ([1, 1], [3, 4, 6, 3]),
     "resnet50" => ([1, 1, 4], [3, 4, 6, 3]),
     "resnet101" => ([1, 1, 4], [3, 4, 23, 3]),
     "resnet152" => ([1, 1, 4], [3, 8, 36, 3]))

ResNet18() = resnet(basicblock, resnet_config["resnet18"]...)

ResNet34() = resnet(basicblock, resnet_config["resnet34"]...)

ResNet50() = resnet(bottleneck, resnet_config["resnet50"]...)

ResNet101() = resnet(bottleneck, resnet_config["resnet101"]...)

ResNet152() = resnet(bottleneck, resnet_config["resnet152"]...)
