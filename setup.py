#!/usr/bin/env python3

from setuptools import setup, find_packages


install_requires = [
    'numpy',
]

extras_require = {
    'gpu': ['pycuda'],
    'plot': ['matplotlib'],
}

tests_require = [
    'nose',
]

setup(
    name='realtimepork',
    version='0.1',
    author='Dmitri Iouchtchenko',
    author_email='diouchtc@uwaterloo.ca',
    description='PIGS wavefunctions combined with HK semiclassical propagation.',
    license='MIT',
    url='https://github.com/0/realtimepork',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=tests_require,
    packages=find_packages(exclude=['tests', 'tests.*']),
    test_suite='nose.collector',
)
