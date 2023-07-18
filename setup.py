import os
import sys
import re
import io

from setuptools import find_packages
from setuptools import setup

__version__ = '1.0'

REQUIRED_PACKAGES = [
]

setup(
    name='abstraction_machines',
    version=__version__,
    author='Greg Hyde',
    author_email='gregory.m.hyde.th@dartmouth.edu',
    description='abstraction machine',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.8',
)
