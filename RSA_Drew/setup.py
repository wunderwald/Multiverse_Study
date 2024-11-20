from setuptools import setup, find_packages

'''
build: python setup.py sdist bdist_wheel
'''

setup(
    name='rsa_drew',
    version='1.2',
    packages=find_packages(),
    package_data={'rsa_drew': ['*.csv']},
    description='Calculate RSA Synchrony',
    author='Moritz Wunderwald',
    author_email='code@moritzwunderwald.de'
)