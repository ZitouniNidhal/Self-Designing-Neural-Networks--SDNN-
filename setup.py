from setuptools import setup, find_packages

setup(
    name="architectai",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "networkx>=3.0",
        "pydantic>=2.0",
        "pydantic-settings>=2.0",
        "rich>=13.0",
        "click>=8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "architect=architectai.cli.main:main",
        ],
    },
)
