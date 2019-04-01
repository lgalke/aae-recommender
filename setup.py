from setuptools import setup

requirements = [
      'numpy',
      'scipy',
      'sklearn',
      'torch',
      'gensim',
      'pandas',
      'joblib',
      'matplotlib',
      'seaborn',
      'tensorflow-gpu==1.12.0',
]
setup(name='aaerec',
      version=0.1,
      description='Multi-modal Adversarial Autoencoders as Recommender Systems',
      author="Lukas Galke",
      author_email="lga@informatik.uni-kiel.de",
      install_requires=requirements
      )
