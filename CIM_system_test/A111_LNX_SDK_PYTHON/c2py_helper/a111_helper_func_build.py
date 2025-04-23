import sys
from pathlib import Path
import re
from cffi import FFI
ffibuilder = FFI()

base_dir = Path(sys.argv[0]).absolute().parent

# cdef() expects a string listing the C types, functions and
# globals needed from Python. The string follows the C syntax.
ffibuilder.cdef("""
    int a111_helper_test_func(int i);
""")

# This describes the extension module "a111_helper_cffi" to produce.
ffibuilder.set_source("a111_helper_cffi",
"""
     #include "a111_helper.h"   // the C header of the library
""",
     libraries=['a111_helper'],    # library name, for the linker
     library_dirs=[str(base_dir)], 
     include_dirs=[str(base_dir)])   

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

