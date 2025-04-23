from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'a111 sdk for python3'
LONG_DESCRIPTION = 'this is a python3 wrapper for a111 sdk, used in jupyter notebook'

setup(
        name="a111sdk", 
        version=VERSION,
        url='www.ime.tsinghua.edu.cn',
        license='GPL',
        author="Liu Hang",
        author_email="<liuhang_icfc@tsinghua.edu.cn>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        # install_requires=['numpy'], # add any additional packages that 
        include_package_data=True,

        
        keywords=['python3', 'a111 sdk'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: TSINGHUA EDU",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: PYNQ",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)