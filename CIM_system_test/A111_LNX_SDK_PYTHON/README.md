# a111sdk Python Package Installation and Usage Guide

## 1. Package Generation

### Method 1: Generate a .whl Installation File
1. Navigate to the `a111sdk-pack` directory.
2. Run the command `python3 setup.py bdist_wheel`.
3. A `.whl` installation file will be generated in the `dist` directory.

### Method 2: Editable Installation
1. Navigate to the `a111sdk-pack` directory.
2. Run the command `pip3 install -e .`.
3. A `.egg_info` file will be generated in the current directory.
4. Navigate to the `/usr/local/lib/python3.7/dist-packages` directory.
5. Create a file named `easy_install.pth`.
6. Open the `easy_install.pth` file and enter the absolute path of the `a111sdk-pack` directory.
7. Save the file and exit.

## 2. Installation Testing
1. Navigate to the user's home directory.
2. Start the Python shell by running `sudo python3`.
3. Inside the Python shell, run `import sys` and then `print(sys.path)`.
4. If the output includes the absolute path of the `a111sdk-pack` directory, the environment path has been added successfully.
5. Try to import the `a111sdk` package by running `import a111sdk` in the Python shell.

**Note**: After updating the source code, you need to repeat the entire process for Method 1. For Method 2, there's no need to repeat the process, but avoid changing the source code file path.

## 3. Running and Testing the Samples
### Prerequisites
All Jupyter notebook sample files are located in the `notebooks` directory.

### Steps
1. Start the Jupyter notebook server by running `sudo jupyter notebook --allow-root`.
2. Open a web browser and enter the device IP address followed by `:8888`.
3. Enter the password `icfc` when prompted.
4. You will see all the sample notebooks and can start exploring and testing.