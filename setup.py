from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='fusion_kan',
    version='1.0.0',
    author='Researcher',
    description='High-Performance CUDA implementation of Kolmogorov-Arnold Networks',
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=['fusion_kan'],
    ext_modules=[
        CUDAExtension(
            name='_fusion_kan_cuda',
            sources=['csrc/fusion_kan.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O3', '--use_fast_math']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch', 'numpy']
)