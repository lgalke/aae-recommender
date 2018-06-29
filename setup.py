from setuptools import setup
import os
reqfile = os.path.join(os.path.dirname(__file__), 'requirements.txt')
with open(reqfile, 'r') as f:
      requirements = [line.strip() for line in f.readlines()]

setup(name='aaerec',
      version=0.1,
      description='Multi-modal Adversarial Autoencoders as Recommender Systems',
      author="Lukas Galke",
      author_email="lga@informatik.uni-kiel.de",
      install_requires=requirements
      )
