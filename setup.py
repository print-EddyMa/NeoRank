from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neorank",
    version="1.0.0",
    author="Eddy Ma",
    author_email="your-email@example.com",
    description="Accessible neoantigen immunogenicity prediction using peptide sequences and HLA typing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/NeoRank",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
