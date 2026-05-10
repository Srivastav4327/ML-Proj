from setuptools import  find_packages,setup
from typing import List
HYPEN_E_DOT = '-e.'
def get_requirements(file_path:str)->List[str]:
    
    '''
    this function will return the list of requirements
     from the given file path
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]#to igonre spaces and new line characters
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='ML-Proj',
    version='0.0.1', 
    author='Srivastav',
    author_email='srisrivastav863@gmail.com',
    packages=find_packages(),
    #|find_packages() will automatically find all the packages in the project and include them in the distribution.
    install_requires=get_requirements('requirements.txt')

)
