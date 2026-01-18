from setuptools import setup, find_packages

setup(
    name="ml-timeseries",
    version="0.1.0",
    description="End-to-end time series forecasting framework",
    author="Anurag Pandey",
    author_email="anurag.pandey1292@gmail.com",
    url="https://github.com/anuragpandey1292/time_series_model",
    license="MIT",

    # IMPORTANT for src layout
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    python_requires=">=3.8",

    # KEEP THIS MINIMAL
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "pyyaml>=6.0",
    ],

    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.950",
        ],
        "api": [
            "fastapi>=0.75.0",
            "uvicorn>=0.17.0",
            "gunicorn>=20.1.0",
        ],
        "dl": [
            "tensorflow>=2.8.0",
            "torch>=1.10.0",
            "pytorch-lightning>=1.6.0",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    include_package_data=True,
    zip_safe=False,
)
