from setuptools import setup, find_packages

setup(
  name="stagpyviz",
  version="1.0.0",
  packages=find_packages(include=["stagpyviz", "stagpyviz.*"]),
  install_requires=[
    "numpy",
    "scipy",
    "pyvista",
    "pint"
  ],
  author="Anthony Jourdon",
  description="A Python package for visualizing StagYY Yin-Yang models",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/anthony-jourdon/stagpyviz",
  classifiers=[
    "Programming Language :: Python :: 3"
  ],
  python_requires=">=3.13"
)