"""Setup configuration for instruction_hierarchy package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="instruction-hierarchy",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Mechanistic interpretability of instruction hierarchy in language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/instruction-hierarchy",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "einops>=0.7.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "huggingface-hub>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "jupyter": [
            "ipython>=8.12.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "quantization": [
            "bitsandbytes>=0.41.0",
        ],
    },
)
