from setuptools import setup, find_namespace_packages

setup(
    name="dataset_quality_checker",
    version="0.1.0",
    description="A comprehensive data quality analysis tool",
    author="Navdeep Gill",
    packages=find_namespace_packages(
        include=["dataset_quality_checker*", "thinc*"],  # Adjust to your package structure
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    install_requires=[
        "numpy>=1.21.0",                    # Ensure compatibility with Python 3.13
        "scipy>=1.8.0",                     # Update dependencies as necessary
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.7",
        "langdetect>=1.0.9",
        "textblob>=0.17.1",
        "phonenumbers>=8.12.0",
        "gensim>=4.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "spacy>=3.3.0",
        "ruptures>=1.1.0",
        "rake-nltk>=1.0.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",                # For unit testing
            "black>=22.1.0",               # Code formatting
            "flake8>=5.0.0",               # Code linting
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",  # Explicitly declare support
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
