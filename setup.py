#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel
# @File    ：setup.py
# @Author  ：sl
# @Date    ：2022/4/10 13:46
from __future__ import print_function

import sys
from setuptools import setup, find_packages

PROJECT_NAME = "nlptravel"

__version__ = None
exec(open(f'{PROJECT_NAME}/version.py').read())

if sys.version_info < (3,):
    sys.exit(f'Sorry, Python3 is required for {PROJECT_NAME}.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name=PROJECT_NAME,
    version=__version__,
    description='Nlp travel',
    # long_description=readme,
    long_description=PROJECT_NAME,
    license='Apache License 2.0',
    url='https://github.com/CycloneBoy/nlptravel',
    author='CycloneBoy',
    author_email='cycloneboy@foxmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    keywords='NLP,Chinese,Travel,pytorch,deep learning',
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(),
    # package_data={
    #     'nlptravel': ['*.*', '../LICENSE', '../README.*', '../*.txt', 'data/*', 'data/en/en.json.gz',
    #                     'utils/*.', 'bert/*', 'deep_context/*', 'conv_seq2seq/*', 'seq2seq_attention/*',
    #                     'transformer/*', 'electra/*']}
)
