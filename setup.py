from setuptools import setup

REQUIRED_PACKAGES = [
    "attrs == 18.2.0",
    "cattrs == 0.9.0",
    "pycocotools == 2.0.0",
]

setup(name='suppose',
      version='0.1',
      description='Superfast Utilities for Position and Pose Orientation Sequence Estimation',
      url='https://github.com/WildflowerSchools/suppose',
      author='Lue Andrie-Her',
      author_email='lue.her@gmail.com',
      license='MIT',
      packages=['suppose'],
      zip_safe=True,
      setup_requires=[],
      scripts=['bin/suppose'],
      install_requires=REQUIRED_PACKAGES,
)