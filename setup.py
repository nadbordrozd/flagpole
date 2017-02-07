#!/usr/bin/env python

from distutils.core import setup

LICENSE = open("LICENSE").read()

# strip links from the descripton on the PyPI
LONG_DESCRIPTION = open("README.md")


setup(name='aimia-dstk',
   version='0.11',
   description='buncha utils for data science',
   long_description=LONG_DESCRIPTION,
   author='Nadbor Drozd',
   author_email='nadbor.drozd@aimia.com',
   url='https://github.com/nadbor-aimia/ds-tools',
   license=LICENSE,
   classifiers= [ "Development Status :: 3 - Alpha",
                  "License :: OSI Approved :: MIT License",
                  "Operating System :: OS Independent",
                  "Programming Language :: Python :: 2.7",
                  "Topic :: Software Development :: Libraries" ],
   packages = ['dstk', 'dstk.imputation'])