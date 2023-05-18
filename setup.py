from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='atomInSmiles',
      version='1.0.2',
      description='Atom-in-SMILES tokenizer for SMILES strings',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Islambek Ashyrmamatov, Umit V. Ucak,  Juyong Lee',
      author_email='ashyrmamatov01@gmail.com', #azpisruh@gmail.com,  drfaust23@gmail.com}',
      license='Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)',
      url='https://github.com/snu-lcbc/atom-in-SMILES',
      packages=['atomInSmiles'],
      install_requires=[
          'rdkit>=2022.9.5',
      ],
      python_requires='>=3.8',
      zip_safe=True
)
