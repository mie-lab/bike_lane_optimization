"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ""
if os.path.exists("README.md"):
    with open("README.md") as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name="ebike_city_tools",
    version="0.0.1",
    description="Bike lane network optimization",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="MIE Lab",
    author_email=("nwiedemann@ethz.ch"),
    license="MIT",
    url="https://github.com/mie-lab/bike_lane_optimization",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "networkx",
        "matplotlib",
        "seaborn",
        "rdflib"
    ],
    classifiers=[
        "License :: OSI Approved :: MIT",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages("."),
    python_requires=">=3.9",
    scripts=scripts,
)
