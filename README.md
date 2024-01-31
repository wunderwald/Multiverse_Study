# MULTIVERSE STUDY

This repository contains all code that is part of the ECG synchrony multiverse study.

## PACKAGES

### Concepts

Conceptual texts about algorithms.

### IBI_Generator

HRV simulation based inter-beat interval sequence generation, semi-random physiology-based parameter picking.

### RSA_Drew

Optimized python transcript of Drew Abbney's RSA synchrony algorithm.

### ML

Machine learning algorithm for predicting parameters of the IBI_Generator that lead to high synchrony scores in RSA_Drew

## PACKAGE_MANAGEMENT

Each package is a standalone working unit but ML requires code from IBI_Generator and RSA_Drew. Whenever the latter packages are changed, they must be **built and re-installed in the ML package**.

### Building

In the directories IBI_Generator or RSA_Drew, run `python setup.py sdist bdist_wheel` (with venv deactivated).

### Adding Exports

The functions to be included in the build are defined in `__init__.py` (in the package directories of IBI_Generator and RSA_Drew)

### Installing

In the ML package, install the builds of the other packages using `pip install ../RSA_Drew/dist/rsa-0.4-py3-none-any.whl` (for example)