include("GilaOperators.jl")
include("spheres.jl")
using ..GilaOperators

# Create a domain for the (I-XG) operator to act on
cells = [32, 32, 1] # Number of cells in each direction (we can simulate 2D by setting one dimension to 1)
scale = (1//100, 1//100, 1//100) # Each cell is 0.01x0.01x0.01 wavelengths big
coord = (0//1, 0//1, 0//1) # The cells are centered at (0, 0, 0)
chis = [-10+0.1im, -3+0.1im, 3+0.1im, 10+0.1im] # Some list of possible susceptibilities (negative real part: metal, positive real part: dielectric, imaginary part: loss)
num_spheres = 7 # Some random number
medium = create_random_medium(cells, num_spheres, chis) # The X operator. Note that it might be more interesting to have different media other than spheres to train the network on

use_gpu = false

# Create a LippmannSchwinger operator (the (I-XG) operator)
ls = LippmannSchwinger(cells, scale, coord, medium; use_gpu=use_gpu)

# Create some sources for the operator to act on
vec = rand(ComplexF64, (cells..., 3))
out = ls * vec # This is the result of the (I-XG) operator acting on vec. This is the data you want to store to train the network
