from setuptools import setup, find_packages

with open("README.md") as f:
    LONG_DESC = f.read()

setup(
    name="percolation",
    version=0.1,
    description="Simple percolation model based on a square lattice",
    author="Joe Marsh Rossney",
    url="https://github.com/marshrossney/percolation",
    long_description=LONG_DESC,
    package=find_packages(),
    entry_points={
        "console_scripts": [
            "perc-anim = percolation.scripts.shell_scripts:anim",
            "perc-time = percolation.scripts.shell_scripts:time",
            "perc-scan = percolation.scripts.shell_scripts:scan",
        ]
    },
)
