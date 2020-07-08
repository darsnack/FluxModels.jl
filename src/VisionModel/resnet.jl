using Flux

basicblock(inplanes::Int, outchannels, downsample::Bool = false) = downsample ? 
  Chain(Conv((3, 3), inplanes => outchannels[1], stride = 2),
        BatchNorm(inplanes, λ = relu),
        Conv((3, 3), inplanes => outchannels[2], stride = 1, pad = 1),
        BatchNorm(outchannels, λ=relu)) : 
  Chain(Conv((3, 3), inplanes => outchannels[1], stride = 1),
        BatchNorm(inplanes, λ = relu),
        Conv((3, 3), inplanes => outchannels[2], stride = 1, pad = 1),
        BatchNorm(outchannels, λ=relu))

bottleneck(inplanes::Int, outchannels, downsample::Bool = false) = downsample ?
  Chain(Conv((1, 1), inplanes => outchannels[1], stride = 2),
        BatchNorm(inplanes, λ = relu),
        Conv((3, 3), inplanes => outchannels[2], stride = 1, pad = 1),
        BatchNorm(inplanes, λ = relu),
        Conv((1, 1), inplanes => outchannels[3], stride = 1),
        BatchNorm(outchannels, λ = relu)) :
  Chain(Conv((1, 1), inplanes => outchannels[1], stride = 1),
        BatchNorm(inplanes, λ = relu),
        Conv((3, 3), inplanes => outchannels[2], stride = 1, pad = 1),
        BatchNorm(inplanes, λ = relu),
        Conv((1, 1), inplanes => outchannels[3], stride = 1),
        BatchNorm(outchannels, λ = relu))

projection(inplanes::Int, outplanes::Int, stride::Int) = Chain(Conv((1, 1), inplanes => outplanes, stride=stride),
                                                               BatchNorm((outplanes), λ=relu))

identity(inplanes::Int, outplanes::Int, stride::Int) = +

function resnet(block, shortcut_config, channel_config, block_config)
  layers = []
  push!(layers, Conv((7, 7), 3=>inplanes, stride=(2, 2), pad=(3, 3)))
  push!(layers, BatchNorm(inplanes, λ=relu))
  push!(layers, MaxPool((3, 3), stride=(2, 2), pad=(1, 1)))
  inplanes = 64
  baseplanes = 64
  for nrepeats in block_config
    outplanes = baseplanes .* channel_config
    # downsample the first block
    if shortcut_config == :A
      # zero padded identity shortcut
    elseif shortcut_config == :B || shortcut_config == :C
      push!(layers, SkipConnection(block(inplanes, outplanes, true),
                                   projection(inplanes, outplanes[end], 2)))
    end
    inplanes = outplanes[end]
    for i in 1:nrepeats
      if shortcut_config == :A || shortcut_config == :B
        push!(layers, SkipConnection(block(inplanes, outplanes, false),
                                     identity(inplanes, outplanes[end], 2)))
      elseif shortcut_config == :C
        # projection shortcut
      end
      inplanes = outplanes[end]
    end
    baseplanes *= 2
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

ResNet18() = resnet(basicblock, ":A", resnet_config["resnet18"]...)

ResNet34() = resnet(basicblock, ":A", resnet_config["resnet34"]...)

ResNet50() = resnet(bottleneck, ":B", resnet_config["resnet50"]...)

ResNet101() = resnet(bottleneck, ":B", resnet_config["resnet101"]...)

ResNet152() = resnet(bottleneck, ":B", resnet_config["resnet152"]...)
