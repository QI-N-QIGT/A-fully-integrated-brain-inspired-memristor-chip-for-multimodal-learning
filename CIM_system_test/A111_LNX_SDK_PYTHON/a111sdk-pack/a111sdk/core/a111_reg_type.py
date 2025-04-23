def _ns_get(key, namespace, clses, dft=None):
    try:
        return namespace[key]
    except KeyError:
        pass
    for cls in clses:
        try:
            return getattr(cls, key)
        except AttributeError:
            pass
    return dft


class RegType(type):

    def __new__(cls, clsname, bases, namespace):
        fields = _ns_get('FIELDS', namespace, bases)
        nbytes = _ns_get('NBYTES', namespace, bases)
        if fields is not None and nbytes is not None:
            nbits = 0
            names = {}
            total = nbytes * 8
            for name, bits in fields:
                if bits == 0:
                    bits = total - nbits
                assert 0 < bits <= total, f'invalid field {name} bits = {bits}'
                assert nbits + bits <= total, f'field {name} overflowed'
                if name != '_':
                    assert name not in names, f'field {name} duplicated'
                    names[name] = (nbits, bits)
                nbits += bits
            assert 'bits' not in names, 'field \"bits\" is invalid'
            namespace.update(__slots__=('bits', *names), _FIELDS=names)
        return type.__new__(cls, clsname, bases, namespace)


class BaseReg(metaclass=RegType):

    FIELDS = None
    NBYTES = None

    def __init__(self, bits=0, **kwargs):
        self.bits = bits
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        pos, bits = self._FIELDS[name]
        return (self.bits >> pos) & ((1 << bits) - 1)

    def __setattr__(self, name, value):
        if name in ('bits',):
            return super().__setattr__(name, value)
        pos, bits = self._FIELDS[name]
        assert 0 <= value < (1 << bits), f'field {name} = {value} overflows'
        mask = ((1 << bits) - 1) << pos
        self.bits = (self.bits & ~mask) | (value << pos)

    def to_bytes(self, big_endian=False):
        return self.bits.to_bytes(self.NBYTES,
                                  'big' if big_endian else 'little')

    def from_bytes(self, value, big_endian=False):
        assert len(value) == self.NBYTES, \
                f'bytes {value} length != {self.NBYTES}'
        self.bits = int.from_bytes(value, 'big' if big_endian else 'little')

    def bits_of(self, name):
        return self._FIELDS[name][1]


class Reg32(BaseReg):
    NBYTES = 4
