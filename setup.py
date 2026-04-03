from setuptools import setup, find_packages

setup(
    name="kopeer-lib",  # nombre único en PyPI
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0"
    ],
    python_requires='>=3.10',
    description="Librería para emparejamiento de tutoría entre pares usando grafos y scores vectorizados",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Favio Asturimac",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)