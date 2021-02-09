from setuptools import setup, find_packages

with open("README.md") as f:
    LONG_DESC = f.read()

setup(
    name="p1b-pandemic",
    version=0.1,
    description="Basic pandemic model based on a square lattice",
    author="Joe Marsh Rossney",
    url="https://github.com/marshrossney/p1b-pandemic",
    long_description=LONG_DESC,
    package=find_packages(),
    entry_points={
        "console_scripts": [
            "p1b-evolve = p1b_pandemic.scripts.shell_script:main",
            "p1b-scan = p1b_pandemic.scripts.parameter_scan:main",
        ]
    },
)
