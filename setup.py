from typing import List
from setuptools import find_packages,setup



def get_requirement(file_path:str)->list[str]:
    requirements=[]
    with open(file_path,encoding='utf-8') as obg_require:
            requirements=obg_require().readlines()
            requirements= [req.strip() for req in requirements if req.strip() and not req.startswith('-e')]
    return requirements

setup(
    name="customersEcommerce",
    author="MohamedNasser",
    author_email="mohamednasserabohamda",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirement('requirement.txt')
)