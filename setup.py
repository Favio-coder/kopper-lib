from setuptools import setup, find_packages

# Leer README con encoding correcto
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kopeer-lib",
    version="0.2.1",  # Actualiza versión para PyPI
    packages=find_packages(exclude=["tests", "tests.*"]),
    
    # Dependencias principales
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
    ],
    
    # Dependencias opcionales (para desarrollo/testing)
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "build>=0.10.0",
            "twine>=4.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    
    python_requires='>=3.8',  # Más compatible
    
    description="Librería para emparejamiento de tutoría entre pares usando redes neuronales ligeras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    author="Favio Asturimac",
    author_email="faviusam@gmail.com",  # Agrega email
    url="https://github.com/Favio-coder/kopeer-lib",
    
    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Incluir archivos adicionales
    include_package_data=True,
    zip_safe=False,
    
    # Palabras clave para búsqueda en PyPI
    keywords="peer-tutoring, education, machine-learning, neural-networks, recommendations",
    
    # Licencia
    license="MIT",
)