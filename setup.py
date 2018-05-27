from setuptools import setup, find_packages

setup(
    name='pymad8',
    version='1.3',
    packages=find_packages(exclude=["docs", "tests", "old"]),
    # Not sure how strict these need to be...
    install_requires=["matplotlib >= 1.7.1",
                      "numpy >= 1.3.0",
                      "fortranformat >= 0.2.5"],
    # Some version of python2.7
    python_requires="==2.7.*",

    author='JAI@RHUL',
    author_email='stewart.boogert@rhul.ac.uk',
    description="Write MAD8 models and load MAD8 output.",
    url='https://bitbucket.org/jairhul/pymad8/',
    license='GPL3',
    keywords='mad8 accelerator twiss'
)
