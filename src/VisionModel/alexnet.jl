function alexnet()
  layers = Chain(Conv((11, 11), 3=>64, stride=(4, 4), relu, pad=(2, 2)),
                 MaxPool((3, 3), stride=(2, 2)),
                 Conv((5, 5), 64=>192, relu, pad=(2, 2)),
                 MaxPool((3, 3), stride=(2, 2)),
                 Conv((3, 3), 192=>384, relu, pad=(1, 1)),
                 Conv((3, 3), 384=>256, relu, pad=(1, 1)),
                 Conv((3, 3), 256=>256, relu, pad=(1, 1)),
                 MaxPool((3, 3), stride=(2, 2)),
                 AdaptiveMeanPool((6,6)),
                 flatten,
                 Dropout(0.5),
                 Dense(256 * 6 * 6, 4096, relu),
                 Dropout(0.5),
                 Dense(4096, 4096, relu),
                 Dense(4096, 1000))
  Flux.testmode!(layers, false)
  return layers
end
