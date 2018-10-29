from setuptools import setup

REQUIRED_PACKAGES = [
    'Click >= 7.0, < 8',
    'Logbook >= 1.4.1, < 2',
    'numpy >= 1.14.5',
    'pandas >= 0.23.4',
    'tqdm >= 4.26.0, < 5',
    "tf-pose == 0.1.1",
]

DEPENDENCY_LINKS = [
    'https://github.com/ildoonet/tf-pose-estimation/tarball/master#egg=tf-pose-0.1.1',
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
      setup_requires=['numpy'],
      scripts=['bin/suppose'],
      install_requires=REQUIRED_PACKAGES,
      dependency_links=DEPENDENCY_LINKS
)