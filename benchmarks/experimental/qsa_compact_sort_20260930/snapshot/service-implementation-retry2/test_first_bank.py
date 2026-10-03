"""Byte-backed CPU mock verifies bank lifecycle, not torch/CUDA compatibility."""
import itertools
import struct
import unittest
from types import SimpleNamespace as NS
from first_failure import FirstFailure


DTYPES = {'float16': ('H', 2), 'int16': ('h', 2), 'float32': ('I', 4),
          'uint32': ('I', 4), 'int32': ('i', 4), 'int64': ('q', 8)}


class Tensor:
    def __init__(self, shape, dtype, raw=None, indices=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, 'cpu_mock'
        self.raw = raw if raw is not None else bytearray(self.numel() * self.element_size())
        self.indices = list(range(self.numel())) if indices is None else indices

    def numel(self):
        count = 1
        for n in self.shape: count *= n
        return count

    def element_size(self): return DTYPES[self.dtype][1]
    def values(self):
        fmt, width = DTYPES[self.dtype]
        return [struct.unpack('<' + fmt, self.raw[i * width:(i + 1) * width])[0]
                for i in self.indices]
    def bytes(self):
        width = self.element_size()
        return b''.join(self.raw[i * width:(i + 1) * width] for i in self.indices)
    def cpu(self): return self
    def tolist(self): return self.values()
    def __eq__(self, value): return len(self.indices) == 1 and self.values()[0] == value
    def __getitem__(self, key):
        if isinstance(key, int): return Tensor((), self.dtype, self.raw, [self.indices[key]])
        ranges = [list(range(self.shape[i]))[part] for i, part in enumerate(key)]
        strides = [1] * len(self.shape)
        for i in range(len(self.shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]
        indices = [self.indices[sum(k * s for k, s in zip(coords, strides))]
                   for coords in itertools.product(*ranges)]
        return Tensor(tuple(len(x) for x in ranges), self.dtype, self.raw, indices)
    def view(self, dtype): return Tensor(self.shape, dtype, self.raw, self.indices)
    def copy_(self, source):
        payload = source.bytes(); width = self.element_size()
        for j, i in enumerate(self.indices):
            self.raw[i * width:(i + 1) * width] = payload[j * width:(j + 1) * width]
        return self
    def zero_(self):
        self.raw[:] = bytes(len(self.raw)); return self


class Torch:
    float16, float32, int16, int32, int64, uint32 = tuple(DTYPES)
    contiguous_format = None
    @staticmethod
    def empty_like(value, **kwargs): return Tensor(value.shape, value.dtype)
    @staticmethod
    def zeros(size, dtype, **kwargs): return Tensor((size,), dtype)
    @staticmethod
    def tensor(values, dtype, **kwargs):
        tensor = Tensor((len(values),), dtype)
        fmt = DTYPES[dtype][0]
        tensor.raw[:] = b''.join(struct.pack('<' + fmt, value) for value in values)
        return tensor
    @staticmethod
    def where(take, a, b): return a if take else b


def input_set(bits):
    q = Tensor((2, 2), 'float16')
    q.raw[:] = b''.join(struct.pack('<H', b) for b in bits)
    p = Torch.tensor([1, 2], 'int32')
    return {'q': q, 'reference_pages': p}


class Lifecycle(unittest.TestCase):
    def test_false_first_second_failure_drain_reset_bits_and_alias(self):
        first = input_set([0x0000, 0x8000, 0x7E55, 0x7F31])
        second = input_set([0x3C00, 0xBC00, 0x7EAA, 0x7F99])
        bank = FirstFailure(Torch, first, 64 * 1024**2)
        bank.prepare(0, first); bank.prepare(1, second)
        self.assertIsNot(bank.bank['q'].raw, first['q'].raw)
        self.assertIsNot(bank.bank['reference_pages'].raw, first['reference_pages'].raw)
        original = first['q'].bytes(), first['reference_pages'].bytes()
        before = bank.bank['q'].bytes()
        bank.capture(0, first, False)
        self.assertEqual(bank.bank['q'].bytes(), before)
        bank.capture(0, first, True)
        saved = bank.bank['q'].bytes()
        self.assertEqual(saved, first['q'].bytes())
        bank.capture(1, second, True)
        self.assertEqual(bank.bank['q'].bytes(), saved)
        self.assertEqual(bank.flag.tolist(), [1, 0, 2, 2])
        # Safe CPU drain read leaves saved bits/flag unchanged.
        self.assertEqual(bank.bank['q'].cpu().bytes(), saved)
        self.assertEqual(original, (first['q'].bytes(), first['reference_pages'].bytes()))
        bank.reset(); self.assertEqual(bank.flag.tolist(), [0, 0, 0, 0])
        bank.capture(1, second, True)
        self.assertEqual(bank.bank['q'].bytes(), second['q'].bytes())
        self.assertEqual(bank.flag.tolist(), [1, 1, 2, 2])


if __name__ == '__main__': unittest.main(verbosity=2)
