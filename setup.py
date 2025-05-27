from setuptools import setup, Extension
from setuptools import find_packages
import pybind11
import os

ext_modules = [
    Extension(
        'vp_tree',
        sources=['cpp_extension/vp_tree_bind.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17', '-mavx', '-mfma']
    )
]

setup(
    name='infinitysearch',
    version='0.1.0',
    description='Fermat-based approximate nearest neighbor search with VP-tree backend',
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type='text/markdown',
    author='Antonio Pariente',
    license='Custom Non-Commercial Patent License',
    packages=find_packages(),  # More robust than hardcoding 'infinitysearch'
    ext_modules=ext_modules,
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'tensorflow',
        'scikit-learn',
        'pybind11',
        'optuna'
    ],
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent"
    ],
)
