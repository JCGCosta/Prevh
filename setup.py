from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3.8'
]

setup(
    name='prevhlib',
    version='0.0.9',
    description='Data Mining package implementing the PrevhClassifier algorithm.',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
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
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn']
)
