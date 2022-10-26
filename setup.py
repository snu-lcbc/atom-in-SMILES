from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='atomInSmiles',
      version='1.0.0',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Islambek Ashyrmamatov, Umit V. Ucak,  Juyong Lee',
      author_email='{azpisruh@gmail.com, ashyrmamatov01@gmail.com, drfaust23@gmail.com}',
      license='Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)',
      url='https://github.com/knu-lcbc/atomInSmiles',
      packages=['atomInSmiles'],
      install_requires=[
          'rdkit-pypi==2021.3.4',
      ],
      python_requires='>=3.8',
      zip_safe=True
)