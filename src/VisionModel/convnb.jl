# Simple rewrite of Conv without biases, usefull for convolutional layer followed by BatchNorm
using Flux: @functor, glorot_uniform, expand

struct ConvNB{N,M,F,A}
  σ::F
  weight::A
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

function ConvNB(w::AbstractArray{T,N}, σ = identity; stride = 1, pad = 0,
                    dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return ConvNB(σ, w, stride, pad, dilation)
end

function ConvNB(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
              init = glorot_uniform,  stride = 1, pad = 0, dilation = 1) where N
  ConvNB(init(k..., ch...), σ, stride = stride, pad = pad, dilation = dilation)
end

@functor ConvNB

function (c::ConvNB)(x::AbstractArray)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ = c.σ
  cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(conv(x, c.weight, cdims))
end

function Base.show(io::IO, l::ConvNB)
  print(io, "ConvNB(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::ConvNB{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::ConvNB{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))
