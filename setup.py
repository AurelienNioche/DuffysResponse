from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('cmodule.useful_functions', ['cmodule/useful_functions.pyx'], include_dirs=[np.get_include()]),
]
setup(
    ext_modules=cythonize(extensions), install_requires=['numpy', 'Cython', 'tqdm']
)
