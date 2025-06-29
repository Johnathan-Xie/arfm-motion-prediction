from setuptools import setup, find_packages

setup(
    name="motion_predictor",            # The package name
    version="0.1.0",              # Package version
    description="Motion prediction models",
    packages=find_packages(),     # Auto-detects packages in my_project
    python_requires=">=3.10",      # Adjust to your Python version
    install_requires=[
    ]
)