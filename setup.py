"""JEPA Production Package Setup"""

from setuptools import setup, find_packages

setup(
    name="jepa",
    version="0.1.0",
    description="JEPA-based clip mining for autonomous driving",
    author="Research Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.57.0",
        "pillow==11.3.0",
        "numpy==2.3.3",
        "PyYAML==6.0.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "huggingface-hub==0.35.3",
        "tokenizers==0.22.1",
        "safetensors==0.6.2",
        "tqdm==4.67.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
        ],
    },
)
