from setuptools import setup, find_packages

setup(
    name="dataset_quality_checker",
    version="0.1.0",
    description="A comprehensive data quality analysis tool",
    author="Navdeep Gill",
    packages=find_packages(),
    install_requires=[
        "numpy",                       # Numerical operations
        "scipy",                       # Statistical analysis (e.g., KS test, t-test)
        "statsmodels",                 # Time-series decomposition, VIF calculation
        "scikit-learn",                # ML models, Isolation Forest, LDA
        "nltk",                        # Text preprocessing, tokenization
        "langdetect",                  # Language detection
        "textblob",                    # Text processing, sentiment analysis
        "phonenumbers",                # Phone number validation
        "gensim",                      # Word2Vec for text embeddings
        "pandas",                      # Data manipulation and analysis
        "matplotlib",                  # Plotting and visualization
        "seaborn",                     # Statistical visualizations
        "spacy",                       # Named Entity Recognition (NER)
        "ruptures",                    # Change point detection in time series
        "rake-nltk",                   # Keyword extraction from text
    ],
    extras_require={
        "dev": [
            "pytest",                # For unit testing
            "black",                 # Code formatting
            "flake8",                # Code linting
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
