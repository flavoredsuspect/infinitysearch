from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        'vp_tree',  # C++ extension name
        ['cpp_extension/vp_tree_bind.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17']
    ),
]

setup(
    name='infinitysearch',                 # Library name
    version='0.1.0',                       # Version number
    author='Your Name',                    # Author details
    author_email='youremail@example.com',
    description='Fermat-based nearest neighbor search with VP-tree backend',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',  # README format
    packages=['infinitysearch'],           # The folder containing your code
    ext_modules=ext_modules,               # Include C++ extensions
    install_requires=[                     # Dependencies
        'numpy',
        'scipy',
        'torch',
        'scikit-learn',
        'tensorflow',
        'pybind11',
    ],
    python_requires='>=3.7',               # Ensure compatibility with Python 3.7+
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)

