from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Soft nearest neighbors loss for pytorch"

setup(
    name="soft_nearest_neighbors",
    version=VERSION,
    author="AngusNicolson",
    author_email="<angusjnicolson@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages()
)
