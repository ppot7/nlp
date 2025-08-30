from setuptools import setup, find_packages
setup(
    name='amp_nlp',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your package's dependencies here
        'certifi',
        'numpy',
    ],
    author='Paul J Potocki',
    author_email='ppot7@yahoo.com',
    description='Provides utilities for NLP modelling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ppot7/nlp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
