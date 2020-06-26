using Flux

function basicblock(inplanes::Int, planes::Int, stride::Int, dilation::Int)
  basic =  Chain(
      Conv((3, 3), inplanes=>planes, stride=stride, pad=(1,1), dilation=dilation),
      BatchNorm(planes, λ = relu),
      Conv((3, 3), planes=>planes, stride=stride, pad=(1,1), dilation=dilation),
      BatchNorm(planes, λ=relu)
    );
  return basic
end

function bottleneck(inplanes::Int, planes::Int)
  base_width = 64;
  groups = 1;
  width = Int(planes * (base_width / 64.)) * groups;
  residual =  Chain(Conv((1, 1), inplanes=>width, stride=(2,2)),
          BatchNorm(width, λ=relu),
          Conv((3, 3), width=>width, stride=(2,2), pad=(1,1), dilation=dilation),
          BatchNorm(width, λ=relu),
          Conv((1, 1), width=>planes * expansion, stride=(2,2)),
          BatchNorm(planes * expansion, λ=relu));
  return residual
end 

function projection(inplanes::Int, planes::Int, stride::Int, expansion::Int)
  downsample =  Chain(Conv((1, 1), inplanes=>(planes * expansion), stride=stride),
  BatchNorm((planes * expansion), λ=relu));
  return downsample
end

function resnet(configs::Dict{String, Int})
    begin
    planes = 64;
    inplanes = 64;
    expansion = 1;
    stride = 1;
    end
    layers = [];
    push!(layers, Conv((7, 7), 3=>planes, stride=(2, 2), pad=(3, 3)));
    push!(layers, BatchNorm(planes, λ=relu));
    push!(layers, MaxPool((3, 3), stride=(2, 2), pad=(1, 1)));
    i = 1;
  while i < 4
      stride = 2;
      push!(layers, SkipConnection(basicblock(inplanes, planes, stride, dilation), 
                          projection(inplanes, planes, stride, 1)));
      planes = planes * 2;
      inplanes = planes * expansion;
      expansion = 4;
        for j = 1:i
            push!(layers, SkipConnection(bottleneck(inplanes, planes),
                                    projection(inplanes, planes, stride, 4)));
        end
    end
    push!(layers, AdaptiveMeanPool(1, 1));
    push!(layers, x -> flatten(x, 1));
    push!(layers, Dense(512 * expansion, 1000));

  Flux.testmode!(layers)
  return layers;
end

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