"""setup.py — makes src importable as a package during development."""
from setuptools import find_packages, setup

setup(
    name="nids",
    version="1.0.0",
    packages=find_packages(where="."),
    python_requires=">=3.10",
)
