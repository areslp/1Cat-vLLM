"""Strict tensor-header descriptors for torch ZIP saves; no torch import."""
import collections
from dataclasses import dataclass
import hashlib
import io
import math
import pickle
from pathlib import Path
import zipfile

WIDTHS = {'ByteStorage': 1, 'CharStorage': 1, 'BoolStorage': 1,
          'HalfStorage': 2, 'BFloat16Storage': 2, 'ShortStorage': 2,
          'FloatStorage': 4, 'IntStorage': 4, 'UInt32Storage': 4,
          'DoubleStorage': 8, 'LongStorage': 8, 'UntypedStorage': 1}
DTYPES = {'uint8': 1, 'int8': 1, 'bool': 1, 'float16': 2,
          'bfloat16': 2, 'int16': 2, 'float32': 4, 'int32': 4,
          'uint32': 4, 'float64': 8, 'int64': 8}


@dataclass(frozen=True)
class StorageType:
    name: str


@dataclass(frozen=True)
class Storage:
    key: str
    count: int
    width: int


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple
    element_width: int

    @property
    def bytes(self):
        return math.prod(self.shape) * self.element_width


def rebuild(storage, offset, size, stride, *extra):
    assert isinstance(storage, Storage)
    shape, stride = tuple(size), tuple(stride)
    assert len(shape) == len(stride) and offset >= 0
    assert all(isinstance(n, int) and n >= 0 for n in shape + stride)
    last = offset + sum((n - 1) * s for n, s in zip(shape, stride) if n)
    assert not math.prod(shape) or last < storage.count
    return Tensor(storage, offset, shape, stride, storage.width)


def rebuild_v3(storage, offset, size, stride, requires_grad, hooks, dtype,
               *extra):
    assert isinstance(storage, Storage) and storage.width == 1
    assert dtype in DTYPES
    shape, stride = tuple(size), tuple(stride)
    assert len(shape) == len(stride) and offset >= 0
    assert all(isinstance(n, int) and n >= 0 for n in shape + stride)
    width = DTYPES[dtype]
    last = offset + sum((n - 1) * s for n, s in zip(shape, stride) if n)
    assert not math.prod(shape) or (last + 1) * width <= storage.count
    return Tensor(storage, offset, shape, stride, width)


class HeaderUnpickler(pickle.Unpickler):
    def __init__(self, stream):
        super().__init__(stream)
        self.storages = {}

    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'torch' and name in WIDTHS:
            return StorageType(name)
        if module == 'torch.storage' and name == 'UntypedStorage':
            return StorageType(name)
        if module == 'torch' and name in DTYPES:
            return name
        if module == 'torch._utils' and name == '_rebuild_tensor_v3':
            return rebuild_v3
        if module == 'torch._utils' and name in (
                '_rebuild_tensor', '_rebuild_tensor_v2'):
            return rebuild
        raise ValueError(f'unsupported pickle global {module}.{name}')

    def persistent_load(self, pid):
        assert len(pid) == 5 and pid[0] == 'storage'
        _, dtype, key, location, count = pid
        assert isinstance(dtype, StorageType) and isinstance(count, int)
        assert count >= 0 and isinstance(location, str)
        storage = Storage(str(key), count, WIDTHS[dtype.name])
        assert self.storages.setdefault(storage.key, storage) == storage
        return storage


def stream_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_header(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        records = [name for name in names if name.endswith('/data.pkl')]
        assert len(records) == 1
        record = records[0]
        assert archive.getinfo(record).file_size <= 4 * 1024 * 1024
        loader = HeaderUnpickler(io.BytesIO(archive.read(record)))
        data = loader.load()
        assert isinstance(data, dict)
        prefix = record[:-len('data.pkl')]
        for storage in loader.storages.values():
            entry = archive.getinfo(prefix + 'data/' + storage.key)
            assert entry.file_size == storage.count * storage.width
        total = sum(s.count * s.width for s in loader.storages.values())
        tensors = {key: {'shape': list(value.shape),
                         'stride': list(value.stride),
                         'element_width': value.element_width,
                         'bytes': value.bytes}
                   for key, value in data.items() if isinstance(value, Tensor)}
        return {'meta': data['meta'], 'tensor_headers': tensors,
                'retained_storage_bytes': total,
                'zip_file_bytes': Path(path).stat().st_size}
