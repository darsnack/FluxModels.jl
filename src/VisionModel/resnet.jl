using Flux
# I haven't checked ``downsample`` parameter as it is always performed after every block.
function resnet()
  layers = Chain(
      Conv((7, 7), 3=>64, stride=(2, 2), pad=(3, 3)),
      BatchNorm(64, λ=relu),
      MaxPool((3, 3), stride=(2, 2), pad=(1, 1))
      )
  expansion = 1;
  if groups != 1 || base_width != 64
    error('Basic block only supports groups=1 and base_width=64');
  end
  if dilation > 1
    error("Dilation > 1 not supported in BasicBlock");
  end
  planes = 64;    
  layers = SkipConnection(layers, Chain(
    Conv((3, 3), planes ÷ 2 => planes, stride = (1,1), pad = (1,1), dilation = 1),
    BatchNorm(planes, λ = relu),
    Conv((3, 3), planes => planes, stride = (1,1), pad = (1,1), dilation = 1),
    BatchNorm(planes),
    Sequential(Conv((1, 1), planes ÷ 2 => planes * expansion, stride=(1, 1)),
           BatchNorm(planes * expansion)
    ), x -> relu.(x)
  ))
  expansion = 4;
  base_width = 64;
  groups = 1;
  width = int(planes * (base_width / 64.)) * groups;
  planes = 64;
  for i in 2:4
    layers = SkipConnection(layers, Chain(
      Conv((1, 1), planes*2=>width, stride=(2,2), dilation=0),
      BatchNorm(width, λ=relu),
      Conv((3, 3), width=>width, stride=(2, 2), pad=(1, 1), groups=1, dilation=0),
      BatchNorm(planes, λ=relu),
      Conv((1, 1), width=>planes * expansion, stride=(2,2), dilation=0),
      BatchNorm(planes * expansion),
      Sequential(Conv((1, 1), width=>planes * expansion, stride=(2, 2), dilation=0),
            BatchNorm(planes * expansion)
      ), x -> relu.(x)
      ))
    
  layers = SkipConnection(layers, Chain(AdaptiveMeanPool(1, 1)),
    x -> flatten(x, 1),
    Dense(512 * expansion, 1000));
Flux.testmode!(ls)
return layers;
end
#=
resnet_configs =
Dict("resnet18" => (resnet, [2, 2, 2, 2]),
     "resnet34" => (resnet, [3, 4, 6, 3]),
     "resnet50" => (resnet, [3, 4, 6, 3]),
     "resnet101" => (resnet, [3, 4, 23, 3]),
     "resnet152" => (resnet, [3, 8, 36, 3]))

ResNet18() = resnet(resnet_configs["resnet18"])

ResNet34() = resnet(resnet_configs["resnet34"])

ResNet50() = resnet(resnet_configs["resnet50"])

ResNet101() = resnet(resnet_configs["resnet101"])

ResNet152() = resnet(resnet_configs["resnet152"])
=#