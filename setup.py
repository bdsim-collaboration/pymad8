from setuptools import setup, find_packages

setup(
    name='pymad8',
    version='1.6.1',
    packages=find_packages(exclude=["docs", "tests", "old"]),
    install_requires=["matplotlib>=1.7.1",
                      "numpy>=1.4.0",
                      "fortranformat>=0.2.5"],
    python_requires=">=3.7.*",
    author='JAI@RHUL',
    author_email='stewart.boogert@rhul.ac.uk',
    description="Write MAD8 models and load MAD8 output.",
    license='GPL3',
    url='https://bitbucket.org/jairhul/pymad8/',
    keywords=['mad8', 'accelerator', 'twiss']
)
