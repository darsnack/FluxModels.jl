using Flux

function basicblock()
  dilation = 1;
  if dilation > 1
    error("Dilation > 1 not supported in Basic Block");
  end
  basic =  Chain(
      Conv((3, 3), inplanes=>planes, stride=stride, pad=(1,1), dilation=dilation),
      BatchNorm(planes, λ = relu),
      Conv((3, 3), planes=>planes, stride=stride, pad=(1,1), dilation=dilation),
      BatchNorm(planes)
    );
  return basic
end

function bottleneck(planes::Int, stride::Int)
  stride = 2
  base_width = 64;
  groups = 1;
  width = Int(planes * (base_width / 64.)) * groups;
  residual =  Chain(
          Conv((1, 1), inplanes=>width, stride=(2,2)),
          BatchNorm(width, λ=relu),
          Conv((3, 3), width=>width, stride=(2,2), pad=(1,1), dilation=dilation),
          BatchNorm(width, λ=relu),
          Conv((1, 1), width=>planes * expansion, stride=(2,2)),
          BatchNorm(planes * expansion, λ=relu)
          );
  return residual
end 

function identity(inplanes::Int, planes::Int, stride::Int, expansion::Int)
  inplanes = inplanes;
  planes = planes;
  downsample =  Chain(Conv((1, 1), inplanes=>planes * expansion, stride=stride),
  BatchNorm(planes * expansion, λ=relu));
  return identity
end

function resnet()
  planes = 64;
  inplanes = 64;
  expansion = 1;
  stride = 1;
  layers = Chain(
      Conv((7, 7), 3=>planes, stride=(2, 2), pad=(3, 3)),
      BatchNorm(planes, λ=relu),
      MaxPool((3, 3), stride=(2, 2), pad=(1, 1))
      );
    
  push!(layers, SkipConnection(layers, basicblock(inplanes, planes, stride));
  stride, dilation = basicblock();

  for i in 2:4
    stride = 2;
    push!(layers, SkipConnection(layers, basicblock(inplanes, planes, stride));
    push!(layers, SkipConnection(layers, identity(inplanes, planes, stride, 1)));
    planes = planes*2;
    inplanes = planes * expansion;
    expansion = 4;
      
    for j in 1:i
      push!(layers, SkipConnection(layers, bottleneck(inplanes, planes, dilation)));
      push!(layers, SkipConnection(layers, identity(inplanes, planes, stride, 4)));
      width, dilation = bottleneck();
    end
  
  push!(layers, AdaptiveMeanPool(1, 1))
  push!(layers, x -> flatten(x, 1))
  push!(layers, Dense(512 * expansion, 1000));

  Flux.testmode!(layers)
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