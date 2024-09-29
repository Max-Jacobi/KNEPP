from setuptools import setup

setup(
    name="knepp",
    version="0.1dev",
    author="Max Jacobi",
    packages=["knepp"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "matplotlib",
    ],
    extras_require={
        "WinNet_export": [
            "helmeos @ git+https://github.com/Max-Jacobi/helmeos",
        ],
    },
    python_requires='>=3.10',
    description="KNEC Plotting and Postprocessing",
)
