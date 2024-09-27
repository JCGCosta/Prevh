from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Operating System :: Microsoft :: Windows :: Windows 11',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3.11'
]

setup(
    name='prevhlib',
    version='0.1.6',
    description='Data Mining package implementing the PrevhClassifier algorithm.',
    long_description=
    f"{open('README.md').read()}\n\n" +
    f"{open('CHANGELOG.txt').read()}\n\n",
    long_description_content_type="text/markdown",
    url='https://github.com/JCGCosta/Prevh',
    author='Júlio César Guimarães Costa',
    author_email='juliocesargcosta123@gmail.com',
    license='GNU General Public License v3 (GPLv3)',
    classifiers=classifiers,
    keywords='DataMining',
    #py_modules=["prevh", "Assets"],
    #package_dir={'': 'prevh'},
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'seaborn']
)
