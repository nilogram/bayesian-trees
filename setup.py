from setuptools import setup

setup(
    name="bayesian-trees",
    version="0.0.1",
    description="Bayesian tree-based models for classification tasks.",
    python_requires=">=3.12.12,<3.13",
    packages=["bayesian_trees"],
    include_package_data=True,
    package_data={"bayesian_trees": ["cpp/dirichlet_multinomial_utils.cpp"]},

    # Mandatory native build (runtime compilation via cppimport)
    install_requires=[
        "numpy>=2.4.2,<3.0",
        "pandas>=3.0.1,<4.0",
        "matplotlib>=3.10.8,<4.0",
        "scikit-learn>=1.8.0,<2.0",
        "cppimport>=22.8.2,<23.0",
        "pybind11>=2.12,<3.0",
    ],

    extras_require={
        "examples": [
            "jupyter>=1.1.1,<2.0",
            "ipykernel>=6.29.0,<7.0",
            "ipython>=8.0,<9.0",
        ],
        "pyspark": [
            "pyspark>=4.1.1,<4.2",
            "pyarrow>=23.0.1,<24.0",
        ],
    },
)
