from setuptools import find_packages, setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="zookeeper",
    version="1.3.0",
    author="Plumerai",
    author_email="opensource@plumerai.com",
    description="A small library for managing deep learning models, hyper-parameters and datasets",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/larq/zookeeper",
    packages=find_packages(),
    license="Apache 2.0",
    python_requires=">=3.6",
    install_requires=[
        "click>=7.0",
        "tensorflow-datasets>=1.3.0",
        "typeguard>=2.5.1",
        "importlib-metadata ~= 2.0 ; python_version<'3.8'",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=1.14.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.14.0"],
        "test": [
            "black==20.8b1",
            "docformatter>=1.4",
            "flake8>=3.7.9,<3.10.0",
            "isort==5.8.0",
            "pytest>=4.3.1",
            "pytest-cov>=2.6.1",
            "pytype>=2019.10.17,<2021.5.0",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
)
