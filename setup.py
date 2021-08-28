#!/usr/bin/env python

from pathlib import Path
from setuptools import find_packages
from setuptools import setup


PROJECT_ROOT = Path(__file__).parent


def load_requirements(filename):
    with open(PROJECT_ROOT / filename) as f:
        return f.readlines()


setup(
    name="nncomp_molecule",
    version="0.1.0",
    license="MIT",
    author="Kazuki Fujikawa",
    author_email="k.fujikawa1014@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=load_requirements("requirements.txt"),
    scripts=[],
)
