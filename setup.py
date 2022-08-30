from setuptools import setup

 setup(
   name='fiber-viewer',
   version='0.1.0',
   author='Morgan Hamm',
   packages=['package_name', 'package_name.test'],
   # scripts=[],
   url='https://github.com/MorganHamm/fiber-views/',
   license='LICENSE.txt',
   description='a python package for extracting and manipulating "views" of Fiber-seq data',
   long_description=open('README.md').read(),
   install_requires=[
       "pysam >= 0.19.1",
       "anndata",
   ],
)
