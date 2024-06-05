from fractions import Fraction
from LippmannSchwinger import LippmannSchwinger
import math
import tensorflow as tf
from medium import create_random_medium

def random_complex(shape):
    rs = tf.random.uniform(shape, 0, 1)
    thetas = tf.random.uniform(shape, 0, 2*math.pi)
    return tf.cast(rs, tf.complex128) * tf.exp(1j * tf.cast(thetas, tf.complex128))

# Create a domain for the (I-XG) operator to act on
cells = (32, 32, 1) # Number of cells in each direction (we can simulate 2D by setting the last dimension to 1)
scale = (Fraction(1, 100), Fraction(1, 100), Fraction(1, 100)) # Each cell is 0.01x0.01x0.01 wavelengths big
coord = (Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)) # The cells are centered at (0, 0, 0)
chis = [-10+0.1j, -3+0.1j, 3+0.1j, 10+0.1j] # Some list of possible susceptibilities
num_spheres = 7 # Some random number
medium = create_random_medium(cells, num_spheres, chis)

use_gpu = False # Switch to use the gpu or not

# Create a LippmannSchwinger operator (the (I-XG) operator)
ls = LippmannSchwinger(cells, scale, coord, medium, use_gpu)

# Create a vector for the operator to act on
vec = random_complex((*cells, 3))

# Calculate (I-XG) * vec
out = ls * vec.numpy()
