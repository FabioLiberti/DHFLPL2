from setuptools import setup, find_packages

setup(
    name="dhflpl2",
    version="0.1.0",
    description="Federated Learning in Dynamic and Heterogeneous Environments",
    author="Fabio Liberti, Davide Berardi, Barbara Martini",
    author_email="fabio.liberti@studenti.unimercatorum.it",
    url="https://github.com/FabioLiberti/DHFLPL2",
    license="CC BY 4.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.15.0",
        "flwr>=1.7.0",
        "python-dp>=1.1.5rc4",
        "numpy>=1.24.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
    ],
)
