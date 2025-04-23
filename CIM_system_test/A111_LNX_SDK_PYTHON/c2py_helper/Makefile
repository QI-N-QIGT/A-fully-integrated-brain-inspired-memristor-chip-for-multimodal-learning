CXX := gcc
CXXFLAGS := -Wall
CCOBJFLAGS := $(CXXFLAGS) -c

SDK_H := ./a111_helper.h
SDK_C := ./a111_helper.c
SDK_OBJ := ./a111_helper.o
SDK_A := ./liba111_helper.so
FFI_SO := a111_helper.cpython-37m-aarch64-linux-gnu.so
T_FILES := a111_helper_cffi.c a111_helper_cffi.o
TARGETS := $(FFI_SO)

.PHONY: all clean distclean test

all: $(TARGETS)

clean:
	@rm -rf $(T_FILES) **/__pycache__ $(FFI_SO) $(SDK_OBJ) $(SDK_A)

distclean:
	@rm -rf $(T_FILES) **/__pycache__ $(SDK_OBJ) $(SDK_A)

test: $(TARGETS)
	python3 -c 'from a111_ffi import lib; print(dir(lib));'
	python3 -m unittest discover -v

$(TARGETS): $(SDK_H) $(SDK_A)

$(FFI_SO): a111_helper_func_build.py
	python3 $<
	@rm -rf $(T_FILES) **/__pycache__ $(SDK_OBJ) $(SDK_A)

$(SDK_OBJ):$(SDK_C)
	$(CXX) -c -o $@ $<

$(SDK_A):$(SDK_OBJ)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $(SDK_OBJ)

