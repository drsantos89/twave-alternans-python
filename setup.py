"""Setup file for TWAE."""
from setuptools import find_packages, setup

VERSION = "0.1.0"

DESCRIPTION = (
    "Extract T-wave alternans from ECG signals using the spectral method (K-score)."
)

setup(
    name="twaextractor",
    version=VERSION,
    description=DESCRIPTION,
    author="Diogo Santos",
    author_email="drsantos989@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn",
        "wfdb",
        "matplotlib",
        "scipy",
        "numpy",
        "pydantic",
    ],
    extras_require={
        "dev": [
            "isort",
            "black[jupyter]",
            "flake8",
            "flake8-bugbear",
            "flake8-builtins",
            "flake8-comprehensions",
            "flake8-docstrings",
            "mypy",
            "bandit[toml]",
            "pytest",
            "pytest-cov",
            "pre-commit",
        ],
    },
)
