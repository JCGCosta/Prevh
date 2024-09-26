from setuptools import setup, find_packages
from prevh.Assets.Utils.Static import *

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Operating System :: Microsoft :: Windows :: Windows 11',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3.11.9'
]

setup(
    name='prevhlib',
    version='0.1.1',
    description='Data Mining package implementing the PrevhClassifier algorithm.',
    long_description=
    f"{open('README.md').read()}\n\n" +
    f"{open('CHANGELOG.txt').read()}\n\n" +
    f"Valid Distance Types: {VALID_DISTANCES_TYPES.__repr__()}\n" +
    f"Valid Split Types: {VALID_SPLIT_ALGORITHMS.__repr__()}\n" +
    f"Valid Encoder Algorithms: {VALID_ENCODER_ALGORITHMS.__repr__()}\n" +
    f"Valid Scaler Algorithms: {VALID_SCALER_ALGORITHMS.__repr__()}\n",
    long_description_content_type="text/markdown",
    url='https://github.com/JCGCosta/Prevh',
    author='Júlio César Guimarães Costa',
    author_email='juliocesargcosta123@gmail.com',
    license='GNU General Public License v3 (GPLv3)',
    classifiers=classifiers,
    keywords='DataMining',
    py_modules=["prevh"],
    package_dir={'': 'prevh'},
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'seaborn']
)
