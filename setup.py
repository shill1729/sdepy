from setuptools import setup

setup(
    name='sdepy',
    version='1.12',
    packages=['sdepy'],
    url='https://github.com/shill1729/sdepy',
    requires=["numpy", "sympy", "matplotlib"],
    license='MIT',
    author='Sean Hill',
    author_email='52792611+shill1729@users.noreply.github.com',
    description='Numerical and symbolic solvers for SDEs'
)