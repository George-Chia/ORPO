from setuptools import setup, find_packages

setup(
        name='Anonymous',
        version="0.0.0",
        author='Anonymous',
        author_email='Anonymous',
        maintainer='Anonymous',
        packages=find_packages(),
        platforms=["all"],
        install_requires=[
            "d4rl",
            "gym",
            "matplotlib",
            "numpy",
            "pandas",
            "torch",
            "tensorboard",
            "tqdm",
        ]
    )
