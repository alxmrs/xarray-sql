from setuptools import setup
from Cython.Build import cythonize

setup(
  ext_modules=cythonize(
    ["xarray_sql/**/*_cy.py"],  # Compile all .py files in package
    exclude=["xarray_sql/**/*_test.py"],  # Exclude test files
    compiler_directives={'language_level': "3"},
  ),
)
