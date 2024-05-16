from setuptools import setup, find_packages

setup(
    name='rsa_drew',
    version='1.1',
    packages=find_packages(),
    package_data={'rsa_drew': ['*.csv']},
    description='Calculate RSA Synchrony',
    author='Moritz Wunderwald',
    author_email='code@moritzwunderwald.de'
)