"""
Patched setup.py for ms-pred (ICEBERG).

Original upstream setup.py includes Cython extensions that require compilation.
This patched version removes Cython dependencies and installs as a pure Python
package, which is sufficient for inference-only usage (DAG prediction).

Applied by: pixi run install-msms-iceberg
Source: https://github.com/coleygroup/ms-pred
"""

from setuptools import setup, find_packages

setup(
    name="ms_pred",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "dgl>=0.9.0",
        "rdkit",
        "numpy",
        "scipy",
        "pandas",
        "tqdm",
    ],
)
