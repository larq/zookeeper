from setuptools import find_packages, setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="zookeeper",
    version="1.3.4",
    author="Plumerai",
    author_email="opensource@plumerai.com",
    description="A small library for managing deep learning models, hyper-parameters and datasets",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/larq/zookeeper",
    packages=find_packages(),
    license="Apache 2.0",
    python_requires=">=3.10",
    install_requires=[
        "click>=7.0",
        "tensorflow-datasets>=1.3.0,<v4.9.0",
        "typeguard>=2.5.1,<3.0.0",
        "protobuf<3.21",  # for tensorflow-datasets
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.8.4"],
        "tensorflow_gpu": ["tensorflow-gpu>=2.8.4"],
        "test": [
            "pytype==2024.10.11",
            "pytest==9.0.3",
            "pytest-cov==7.1.0",
            "ruff==0.15.9",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
)
