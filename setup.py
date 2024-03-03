import setuptools

setuptools.setup(
    name='qarray',
    version='0.0.0',
    license='Apache 2.0',
    packages=setuptools.find_packages(exclude=['demo', 'perf_tests']),
    install_requires=['xarray', 'dask-sql'],
    extras_require={
        'dev': ['pyink', 'py-spy'],
        'test': ['zarr', 'gcsfs'],
    },
    python_requires='>=3.9',
)
