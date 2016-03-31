from setuptools import setup

setup(
    name='Spaghetti',
    version='0.0.1',
    description='A Lasagne-compatible conditional random field implementation',
    license='MIT',
    author='Filip Korzeniowski',
    author_email='filip.korzeniowski@jku.at',
    install_requires=['numpy', 'Lasagne', 'Theano']
)
