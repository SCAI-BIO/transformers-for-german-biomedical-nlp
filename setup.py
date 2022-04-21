# -*- coding: utf-8 -*-
import os

from setuptools import find_packages, setup

with os.popen("git describe --tags") as p:
    current_version = p.read().replace("v", "").split("-")[0]
print(f"Installing version {current_version}")

setup(
    name="toolbox",
    version=current_version,
    description="Toolbox for Transformer-assessment",
    author="Manuel Lentzen",
    author_email="manuel.lentzen@scai.fraunhofer.de",
    packages=find_packages(),
    zip_safe=False,
)
install_requires = [
    "torch>=1.6.0",
    "transformers>=4.2.1",
    "mlflow>=1.12.1",
    "pandas>=1.1.4",
    "matplotlib>=3.3.1",
    "scikit-learn>=0.23.2",
    "seqeval>=1.2.2",
]
