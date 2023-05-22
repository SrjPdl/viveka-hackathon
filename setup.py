from setuptools import setup, find_packages  
from typing import List

def get_requirements(filename:str)-> List[str]:
    
    """
    Reads a file containing requirements and returns a list of requirements.\\
    Args:
        filename : The name of the file containing requirements.
    Returns:
        A list of requirements extracted from the file.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an error reading the file.
    Example:
        >>> get_requirements('requirements.txt')
        ['requirement1', 'requirement2', 'requirement3']
    """

    with open(filename) as f:
        return [line.strip() for line in f.readlines() if line.strip() != "-e ."]


setup(name="Viveka-hackathon",
        version="1.0.0",
        description="Viveka-hackathon fake document detection",
        author="Suraj Poudel",
        author_email="075bei044.suraj@pcampus.edu.np",
        packages=find_packages(),
        install_requires=get_requirements("requirements.txt"),
        )