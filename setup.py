from setuptools import setup, find_packages

setup(
    name="cpr-test",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3",
    # setup_requires=['gcsfs==0.7.2'],
    install_requires=[],
    url="https://github.com/stefl14/cpr-test",
    description="A repo for test.",
    include_package_data=True,
)
