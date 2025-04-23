
def twn_n(s):
    assert s >= 1, "twn_n func's input s need tobe >= 1. "
    for k in range(0, 8):
        if (2 ** k) <= s < (2 ** (k+1)):
            if (s / (2 ** k)) <= ((2 ** (k+1)) / s):
                return k
            else:
                return k+1
    return 8

def twn_n_nolimit(s):
    assert s >= 1, "twn_n func's input s need tobe >= 1. "
    k = -1
    while True:
        k += 1
        if (2 ** k) <= s < (2 ** (k+1)):
            if (s / (2 ** k)) <= ((2 ** (k+1)) / s):
                return k
            else:
                return k+1