function place_sphere!(medium::AbstractArray{ComplexF64, 4}, center::AbstractVector{Float64}, radius::Float64, chi::ComplexF64)
    """Places a sphere in the medium with a given susceptibility

    Args:
        medium: 3D array of complex numbers, the susceptibility of the medium at each point
        center: 3-tuple of floats, the center of the sphere in units of cells
        radius: float, the radius of the sphere
        chi: complex number, the susceptibility of the sphere

    Returns:
        medium: 4D array of complex numbers, the susceptibility of the medium at each point
    """
	xs = 1:size(medium, 1)
	ys = 1:size(medium, 2)
	zs = 1:size(medium, 3)
	mask = (xs .- center[1]).^2 .+ (ys .- center[2]).^2 .+ (zs .- center[3]).^2 .< radius^2
	medium[mask, :] .= chi
    return medium
end

function create_random_medium(cells::AbstractVector{Int}, num_spheres::Int, susceptibilities::AbstractVector{ComplexF64})
    """Creates a random medium with a given number of spheres and susceptibilities

    Args:
        cells: 3-tuple of integers, the number of cells in each direction
        num_spheres: int, the number of spheres to place in the medium
        susceptibilities: list of complex numbers, the possible susceptibilities of the spheres

    Returns:
        medium: 4D array of complex numbers, the susceptibility of the medium at each point
    """
	medium = zeros(ComplexF64, cells..., 1)
    for _ in 1:num_spheres
		center = rand(3) .* cells
		radius = rand() * minimum(cells) / 4
		chi = rand(susceptibilities)
        place_sphere!(medium, center, radius, chi)
	end
    return medium
end
