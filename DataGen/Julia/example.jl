include("GilaOperators.jl")
include("spheres.jl")

using Pkg

Pkg.add("CSV")
Pkg.add("DataFrames")
using CSV
using DataFrames

using ..GilaOperators

# Create a domain for the (I-XG) operator to act on
cells = [8, 8, 1] # Number of cells in each direction (we can simulate 2D by setting one dimension to 1)
scale = (1//100, 1//100, 1//100) # Each cell is 0.01x0.01x0.01 wavelengths big
coord = (0//1, 0//1, 0//1) # The cells are centered at (0, 0, 0)
chis = [rp+0.01im for rp=0:0.1:3] # Some list of possible susceptibilities (negative real part: metal, positive real part: dielectric, imaginary part: loss)
num_spheres = 7 # Some random number
#medium = create_random_medium(cells, num_spheres, chis) # The X operator. Note that it might be more interesting to have different media other than spheres to train the network on

use_gpu = false

# Create a LippmannSchwinger operator (the (I-XG) operator)
#ls = LippmannSchwinger(cells, scale, coord, medium; use_gpu=use_gpu)

# Create some sources for the operator to act on
#vec_in = rand(ComplexF64, (cells..., 3))
#out = ls * vec_in # This is the result of the (I-XG) operator acting on vec. This is the data you want to store to train the network

df = DataFrame()

for i in 1:10
    medium = create_random_medium(cells, num_spheres, chis)
    fIn = vec(medium)   
    input_real = convert(Vector{Float64}, real(fIn))
    input_imag = convert(Vector{Float64}, imag(fIn))

    ls = LippmannSchwinger(cells, scale, coord, medium; use_gpu=use_gpu)
    vec_in = rand(ComplexF64, (cells..., 3))
    out = ls * vec_in

    fOut = vec(out)
    output_real = convert(Vector{Float64}, real(fOut))
    output_imag = convert(Vector{Float64}, imag(fOut))

    temp_row = (; 
        [Symbol("input_real_$j") => input_real[j] for j in 1:64]..., 
        [Symbol("input_imag_$j") => input_imag[j] for j in 1:64]...,
        [Symbol("output_real_$j") => output_real[j] for j in 1:192]..., 
        [Symbol("output_imag_$j") => output_imag[j] for j in 1:192]...
    )

    push!(df, temp_row)
end

CSV.write("data.csv", df)

