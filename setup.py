from setuptools import setup, find_packages

setup(
    name="bert_tm",
    version="0.0.0a0",
    description=("Topic Modeling Functionalities"),
    python_requires='>=3.8.12',
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Topic Modeling Functionalities"
    ],
)