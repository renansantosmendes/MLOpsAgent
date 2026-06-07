from setuptools import setup, find_packages

setup(
    name="mlops_agent",
    version="0.1.0",
    description="Production-grade MLOps agent for automated model retraining and deployment",
    author="Renan Santos Mendes",
    author_email="renansantosmendes@example.com",
    url="https://github.com/renansantosmendes/MLOpsAgent",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "joblib>=1.3.0",
        "evidently>=0.4.0",
        "xgboost>=2.0.0",
        "langgraph>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.1.0",
            "black>=23.7.0",
            "mypy>=1.5.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
