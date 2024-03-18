import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
elif os.name == "nt":
    assert False

'''
Usage:

python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)

python setup.py install # build extensions and install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)

python setup.py develop # build extensions and install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)

'''
setup(
    name='raymarching', # package name, import this to use python API
    ext_modules=[
        CUDAExtension(
            name='_raymarching', # extension name, import this to use CUDA API
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'raymarching.cu',
                'bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)