import copy
import os
import multiprocessing
import tensorflow as tf
from juliacall import Main as jl

os.environ["PYTHON_JULIACALL_THREADS"] = str(multiprocessing.cpu_count())
jl.seval("using GilaElectromagnetics, Serialization, LinearAlgebra.BLAS, FFTW")
jl.BLAS.set_num_threads(multiprocessing.cpu_count())
jl.FFTW.set_num_threads(multiprocessing.cpu_count())

class LippmannSchwinger:
    def __init__(self, cells, scale, coord, medium, use_gpu):
        """Implements the operator (I - XG)

        Args:
            cells: 3-tuple of integers, the number of cells in each direction
            scale: 3-tuple of fractions.Fraction (jl.Rational), the size of each cell in units of wavelength
            coord: 3-tuple of fractions.Fraction (jl.Rational), the coordinates of the origin of the domain
            medium: 4D array of complex numbers, the susceptibility of the medium at each point
            use_gpu: bool, whether or not to use the GPU

        Attributes:
            options: Gila.GlaKerOpt, encodes whether or not to use the GPU
            self_volume: Gila.GlaVol, holds information about the domain
            self_mem: Gila.GlaOprMem, holds all the memory the Green's operator needs
        """
        self.options = jl.GlaKerOpt(use_gpu)
        self.self_volume = jl.GlaVol(cells, scale, coord)
        filename = f"../preload/{cells[0]}x{cells[1]}x{cells[2]}_{float(scale[0])}x{float(scale[1])}x{float(scale[2])}@{float(coord[0])},{float(coord[1])},{float(coord[2])}.fur"
        if jl.isfile(filename):
            fourier = jl.deserialize(filename)
            self.self_mem = jl.GlaOprMem(self.options, self.self_volume, egoFur=fourier, setType=jl.ComplexF64)
        else:
            self.self_mem = jl.GlaOprMem(self.options, self.self_volume, setType=jl.ComplexF64)
            jl.serialize(filename, self.self_mem.egoFur)

        # TODO: Convert medium to a CuArray somehow if use_gpu is True
        self.medium = medium

    def __mul__(self, rhs):
        rhs_copy = copy.deepcopy(rhs)
        acted_vec = jl.egoOpr_b(self.self_mem, rhs) # acted_vec = G * rhs
        acted_vec *= tf.expand_dims(self.medium, 3) # acted_vec = XG * rhs
        return rhs_copy - acted_vec # (I - XG) * rhs

