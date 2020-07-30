using Flux

export googlenet

function conv_block(inplanes, outplanes, stride, pad, kernelsize)
  conv_layer = []
  push!(conv_layer, Conv(kernelsize, inplanes => outplanes, stride = stride, pad = pad))
  push!(conv_layer, BatchNorm(outplanes, relu))
  return conv_layer
end

function inception_block(inplanes, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj)
  branch1 = conv_block(inplanes, out_1x1, 0, 0, (1,1))

  branch2 = [conv_block(inplanes, red_3x3, 0, 0, (1,1)),
             conv_block(red_3x3, out_3x3, 0, 1, (3,3))]        

  branch3 = [conv_block(inplanes, red_5x5, 0, 0, (1,1)),
             conv_block(red_5x5, out_5x5, 0, 2, (5,5))] 

  branch4 = [MaxPool((3, 3), stride=1, pad=1),
             conv_block(inplanes, pool_proj, 0, 0, (1,1))]
 
  inception_layer = cat(branch1, branch2..., branch3..., branch4...; dims=1)

  return inception_layer
end

function googlenet()
  layers = Chain(conv_block(3, 64, 2, 3, (7,7)),
                 MaxPool((3,3), stride=2, pad=1),
                 conv_block(64, 64, 0, 0, (1,1)),
                 conv_block(64, 192, 0, 1, (3,3)),
                 MaxPool((3,3), stride=2, pad=1),
                 inception_block(192, 64, 96, 128, 16, 32, 32),
                 inception_block(256, 128, 128, 192, 32, 96, 64),
                 MaxPool((3,3), stride=2, pad=1),
                 inception_block(480, 192, 96, 208, 16, 48, 64),
                 inception_block(512, 160, 112, 224, 24, 64, 64),
                 inception_block(512, 128, 128, 256, 24, 64, 64),
                 inception_block(512, 112, 144, 288, 32, 64, 64),
                 inception_block(528, 256, 160, 320, 32, 128, 128),
                 MaxPool((3,3), stride=2, pad=1),
                 inception_block(832, 256, 160, 320, 32, 128, 128),
                 inception_block(832, 384, 192, 384, 48, 128, 128),
                 AdaptiveMeanPool((1,1)),
                 flatten,
                 Dropout(0.2),
                 Dense(1024, 1000))
  Flux.testmode!(layers, false)
  return layers
end
