# I haven't checked ``downsample`` parameter as it is always performed after every block.
function basicblock(planes::Int)
  expansion = 1;
  if groups != 1 || base_width != 64
    error('basicblock only supports groups=1 and base_width=64');
  end
  if dilation > 1
    error("Dilation > 1 not supported in BasicBlock");
  end
  basic =  Chain(
        Conv((3, 3), planes ÷ 2 => planes, stride = (1,1), pad = (1,1), dilation = 1),
        BatchNorm(planes, λ = relu),
        Conv((3, 3), planes => planes, stride = (1,1), pad = (1,1), dilation = 1),
        BatchNorm(planes),
        Sequential(Conv((1, 1), planes ÷ 2 => planes * expansion, stride=(1, 1)),
               BatchNorm(planes * expansion, λ=relu)
        )
      )
  return basic;
end
# I haven't checked ``downsample`` parameter as it is always performed after every block.
function bottleneck(planes::Int, )
  expansion = 4;
  base_width = 64;
  groups = 1;
  width = int(planes * (base_width / 64.)) * groups
  residual =  Chain(
        Conv((1, 1), planes=>width, stride=(2,2)),
        BatchNorm(planes, λ=relu),
        Conv((3, 3), width=>width, stride=(2, 2), pad=(1, 1), groups=1, dilation=1),
        BatchNorm(planes, λ=relu),
        Conv((1, 1), width=>planes * expansion, stride=(2,2)),
        BatchNorm(planes * expansion, λ=relu),
        Sequential(Conv((1, 1), width=>planes * expansion, stride=(2, 2)),
               BatchNorm(planes * expansion, λ=relu)
        )
      )
  return residual;
end 
function resnet()
  layers = Chain(
      Conv((7, 7), 3=>64, stride=(2, 2), pad=(3, 3)),
      BatchNorm(64, λ=relu),
      MaxPool((3, 3), stride=(2, 2), pad=(1, 1)),
      basicblock(planes)
      )
#= This function has to be completed.
  for i in 2:5
    push!(layers, bottleneck(planes))=#

     

  push!(layers, AdaptiveMeanPool(1, 1));
  push!(layers, x -> flatten(x, 1));
  push!(layers, Dense(512 * expansion, 1000));
  
  end

  # All ReNet functions to be added