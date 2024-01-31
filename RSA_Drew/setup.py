from setuptools import setup, find_packages

setup(
    name='rsa',
    version='0.4',
    packages=find_packages(),
    package_data={'rsa_drew': ['*.csv']},
    description='Calculate RSA Synchrony',
    author='Moritz Wunderwald',
    author_email='code@moritzwunderwald.de'
)