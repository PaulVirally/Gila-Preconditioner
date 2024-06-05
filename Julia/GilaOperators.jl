module GilaOperators

using LinearAlgebra
using Serialization
using Statistics
using GilaElectromagnetics

export GilaOperator, GreensOperator, LippmannSchwinger

abstract type GilaOperator end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, op::GilaOperator, x::AbstractArray{ComplexF64})
	y .= op * x
	return y
end

Base.eltype(::GilaOperator) = ComplexF64

mutable struct GreensOperator <: GilaOperator
	self_mem::GlaOprMem
end

function GreensOperator(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}}, coord::NTuple{3, Rational{Int}}; use_gpu::Bool=false)
	options = GlaKerOpt(use_gpu)
	self_volume = GlaVol(cells, scale, coord)
	filename = "preload/$(cells[1])x$(cells[2])x$(cells[3])_$(float(scale[1]))x$(float(scale[2]))x$(float(scale[3]))@$(float(coord[1])),$(float(coord[2])),$(float(coord[3])).fur"
	if isfile(filename)
		fourier = deserialize(filename)
		self_mem = GlaOprMem(options, self_volume, egoFur=fourier, setType=ComplexF64)
	else
		self_mem = GlaOprMem(options, self_volume, setType=ComplexF64)
		serialize(filename, self_mem.egoFur)
	end
	return GreensOperator(self_mem)
end

function Base.:*(op::GreensOperator, x::AbstractArray{ComplexF64, 4})
	return egoOpr!(op.self_mem, deepcopy(x)) # egoOpr! changes x so we need to deepcopy it
end

function Base.:*(op::GreensOperator, x::AbstractVector{ComplexF64})
	return reshape(egoOpr!(op.self_mem, reshape(deepcopy(x), (op.self_mem.trgVol.cel..., 3))), size(x)) # egoOpr! changes x so we need to deepcopy it
end

mutable struct LippmannSchwinger
	self_mem::GlaOprMem
	medium::AbstractArray{ComplexF64, 4}
end

internal_size(op::LippmannSchwinger) = (op.self_mem.trgVol.cel..., 3)

function Base.:*(op::LippmannSchwinger, x::AbstractVector{ComplexF64})
	x_copy = deepcopy(x)
	acted_vec = egoOpr!(op.self_mem, reshape(deepcopy(x), internal_size(op)))
	acted_vec .*= op.medium
	acted_vec .= reshape(x_copy, size(acted_vec)) .- acted_vec
	return reshape(acted_vec, size(x))
end

function Base.:*(op::LippmannSchwinger, x::AbstractArray{ComplexF64, 4})
	x_copy = deepcopy(x)
	acted_vec = egoOpr!(op.self_mem, x)
	acted_vec .*= op.medium
	acted_vec .= x_copy .- acted_vec
	return acted_vec
end

# Base.eltype(::LippmannSchwinger) = ComplexF64
Base.size(op::LippmannSchwinger) = (prod(op.self_mem.trgVol.cel)*3, prod(op.self_mem.trgVol.cel)*3)
Base.size(op::LippmannSchwinger, ::Int) = prod(op.self_mem.trgVol.cel)*3

function LippmannSchwinger(cells::AbstractVector{Int}, scale::NTuple{3, Rational{Int}}, coord::NTuple{3, Rational{Int}}, medium::AbstractArray{ComplexF64, 4}; use_gpu::Bool=false)
	options = GlaKerOpt(use_gpu)
	self_volume = GlaVol(cells, scale, coord)
	filename = "../preload/$(cells[1])x$(cells[2])x$(cells[3])_$(float(scale[1]))x$(float(scale[2]))x$(float(scale[3]))@$(float(coord[1])),$(float(coord[2])),$(float(coord[3])).fur"
	if isfile(filename)
		fourier = deserialize(filename)
		self_mem = GlaOprMem(options, self_volume, egoFur=fourier, setType=ComplexF64)
	else
		self_mem = GlaOprMem(options, self_volume, setType=ComplexF64)
		serialize(filename, self_mem.egoFur)
	end
	return LippmannSchwinger(self_mem, medium)
end

end
