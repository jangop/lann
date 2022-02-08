"""Locally Adaptive Nearest Neighbors
"""

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="lann",
    version="0.1.0-alpha1",
    description="Locally Adaptive Nearest Neighbors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jangop/lann",
    author="Jan Philip Göpfert",
    author_email="janphilip@gopfert.eu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="machine learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires="~=3.9",
    install_requires=[
        "loguru",
        "scikit-learn",
        "numpy",
        "classicdata",
        "classicexperiments",
    ],
    extras_require={
        "eval": ["pylmnn", "gpyopt", "sklearn-lvq", "matplotlib"],
        "dev": ["check-manifest", "black", "pylint"],
        "test": ["coverage", "pytest", "black", "pylint", "pandas"],
    },
    project_urls={
        "Bug Reports": "https://github.com/jangop/lann/issues",
        "Source": "https://github.com/jangop/lann",
    },
)
