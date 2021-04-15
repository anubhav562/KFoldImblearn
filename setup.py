# todo - do not push this to PyPi until sure or tested by some people

import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="k_fold_imblearn",
    version="1.0.8",
    description="KFoldImblearn handles the resampling of data in a k fold fashion,"
                " taking care of information leakage so that our results are not overly optimistic."
                " It is built over the imblearn package and is compatible with all the oversampling"
                " as well as under sampling methods provided in the imblearn package.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/anubhav562/KFoldImblearn",
    author="Anubhav Chhabra",
    author_email="chhabra.anubhav1997@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    packages=["k_fold_imblearn"],
    include_package_data=True,
    install_requires=[
        "pandas>=1.1.5",
        "scikit-learn>=0.24.0",
        "imbalanced-learn>=0.8.0"
    ]
)
