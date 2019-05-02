from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="larq-swarm",
    version="0.1.0",
    author="Plumerai",
    author_email="lukas@plumerai.co.uk",
    description="A small library for managing deep learning models, hyper parameters and datasets",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/plumerai/larq-swarm",
    packages=find_packages(),
    license="Apache 2.0",
    install_requires=["click>=7.0", "tensorflow-datasets>=1.0.1"],
    extras_require={
        "tensorflow": ["tensorflow>=1.13.1"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.13.1"],
        "test": ["pytest>=4.3.1", "pytest-cov>=2.6.1"],
    },
    entry_points="""
        [console_scripts]
        lq=larq_swarm.main:cli
    """,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
)
