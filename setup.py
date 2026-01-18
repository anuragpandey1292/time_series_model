from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="time-series-forecasting",
    version="1.0.0",
    description="End-to-end time series forecasting and deployment",
    author="Anurag Pandey",
    author_email="anurag.pandey1292@gmail.com",
    url="https://https://github.com/anuragpandey1292/time_series_model",
    license="MIT",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Core dependencies
    install_requires=requirements,
    
    # Optional dependencies for different purposes
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.950",
        ],
        "deployment": [
            "gunicorn>=20.1",
            "flask>=2.0",
            "docker>=5.0",
        ],
    },
    
    # Entry points (if you have CLI commands or apps)
    entry_points={
        "console_scripts": [
            "ts-train=time_series.cli:train",
            "ts-predict=time_series.cli:predict",
            "ts-serve=time_series.cli:serve",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/anuragpandey1292/time_series_model/issues",
        "Source": "https://github.com/anuragpandey1292/time_series_model",
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    include_package_data=True,
    zip_safe=False,
)
