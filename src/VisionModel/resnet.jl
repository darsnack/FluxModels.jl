using Flux

function basicblock()
  dilation = 1;
  if dilation > 1
    error("Dilation > 1 not supported in Basic Block");
  end
  return dilation
end

function bottleneck(planes::Int, stride::Int)
  stride = 2
  base_width = 64;
  groups = 1;
  width = Int(planes * (base_width / 64.)) * groups;
  return width, stride
end 

function identity(inplanes::Int, planes::Int)
  inplanes = inplanes;
  planes = planes;
  return inplanes, planes
end

function resnet()
  planes = 64;
  inplanes = 64;
  expansion = 1;
  layers = Chain(
      Conv((7, 7), 3=>planes, stride=(2, 2), pad=(3, 3)),
      BatchNorm(planes, λ=relu),
      MaxPool((3, 3), stride=(2, 2), pad=(1, 1))
      );
    basic =  Chain(
      Conv((3, 3), inplanes=>planes, stride=(1,1), pad=(1,1), dilation=dilation),
      BatchNorm(planes, λ = relu),
      Conv((3, 3), planes=>planes, stride=(1,1), pad=(1,1), dilation=dilation),
      BatchNorm(planes)
    );
  push!(layers, SkipConnection(basic, basicblock());
  stride, dilation = basicblock();

  for i in 2:4
    push!(layers, SkipConnection(basic, basicblock());
    planes = planes*2;
    inplanes = planes * expansion;
    expansion = 4;
    downsample =  Chain(Conv((1, 1), inplanes=>planes * expansion, stride=stride),
                  BatchNorm(planes * expansion, λ=relu));
      residual =  Chain(
          Conv((1, 1), inplanes=>width, stride=(2,2)),
          BatchNorm(width, λ=relu),
          Conv((3, 3), width=>width, stride=(2,2), pad=(1,1), dilation=dilation),
          BatchNorm(width, λ=relu),
          Conv((1, 1), width=>planes * expansion, stride=(2,2)),
          BatchNorm(planes * expansion, λ=relu)
          );
    for j in 1:i:
      push!(layers, SkipConnection(residual, bottleneck(planes, dilation)));
      push!(layers, SkipConnection(downsample, identity(inplanes, planes)));
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