# -*- coding: utf-8 -*-
# Created by: WU MENGHAO
# Created on: 2019/12/16
from __future__ import print_function
from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="facenet",
    version="1.0.1",
    author="WuMenghao",
    author_email="tm_wmh@foxmail.com",
    description="project use to train facenet and do face validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/WuMenghao/DeepLearningDemo",
    packages=find_packages(),
    install_requires=[
        "tensorflow < 2.4.0",
        ],
    classifiers=[
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
