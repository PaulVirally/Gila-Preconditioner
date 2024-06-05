import numpy as np

def place_sphere(medium, center, radius, chi):
    """Places a sphere in the medium with a given susceptibility

    Args:
        medium: 3D array of complex numbers, the susceptibility of the medium at each point
        center: 3-tuple of floats, the center of the sphere in units of cells
        radius: float, the radius of the sphere
        chi: complex number, the susceptibility of the sphere

    Returns:
        medium: 4D array of complex numbers, the susceptibility of the medium at each point
    """
    for x in range(medium.shape[0]):
        for y in range(medium.shape[1]):
            for z in range(medium.shape[2]):
                if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 < radius**2:
                    medium[x, y, z] = chi
    return medium

def create_random_medium(cells, num_spheres, susceptibilities):
    """Creates a random medium with a given number of spheres and susceptibilities

    Args:
        cells: 3-tuple of integers, the number of cells in each direction
        num_spheres: int, the number of spheres to place in the medium
        susceptibilities: list of complex numbers, the possible susceptibilities of the spheres

    Returns:
        medium: 4D array of complex numbers, the susceptibility of the medium at each point
    """
    medium = np.zeros(cells, dtype=np.complex128)
    rng = np.random.default_rng()
    for _ in range(num_spheres):
        center = rng.random(3) * np.array(cells)
        radius = rng.random() * min(cells) / 4
        chi = rng.choice(susceptibilities)
        medium = place_sphere(medium, center, radius, chi)
    return medium

if __name__ == "__main__":
    cells = (32, 32, 1) # Number of cells in each direction
    chis = [-10+0.1j, -3+0.1j, 3+0.1j, 10+0.1j] # List of possible susceptibilities
    num_spheres = 7 # Some random number
    medium = create_random_medium(cells, num_spheres, chis)

    # Just to get an idea for what the domain looks like
    import matplotlib.pyplot as plt
    from matplotlib import cm
    def chi2color(chi, cmap):
        r = np.real(chi)
        return cmap(np.interp(r, (np.min(r), np.max(r)), (0.0, 1.0)))
    filled = medium != 0.0j
    colors = chi2color(medium, cm.RdBu_r)
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(filled, facecolors=colors, edgecolor='k')
    plt.show()
