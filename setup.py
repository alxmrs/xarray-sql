from setuptools import setup
from Cython.Build import cythonize

import os
import numpy as np
import pyarrow as pa

ext_modules = cythonize(
    ["xarray_sql/**/*_cy.py"],  # Compile all .py files in package
    exclude=["xarray_sql/**/*_test.py"],  # Exclude test files
    compiler_directives={"language_level": "3"},
)

for ext in ext_modules:
  # The Numpy C headers are currently required
  ext.include_dirs.append(np.get_include())
  ext.include_dirs.append(pa.get_include())
  ext.libraries.extend(pa.get_libraries())
  ext.library_dirs.extend(pa.get_library_dirs())

  if os.name == "posix":
    ext.extra_compile_args.append("-std=c++20")

  # Try uncommenting the following line on Linux
  # if you get weird linker errors or runtime crashes
  # ext.define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))

setup(ext_modules=ext_modules)
