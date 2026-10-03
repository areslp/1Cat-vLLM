"""Exact input deployment contract for the unchanged 53-request restore suite."""
from pathlib import Path
from common import BASE, read, sha

FIXTURES = {
    'input-8192.json': '37e8cb0fa41cf30341272b9c0169b41b5205d43481d0ab83f6e3fcdefecd67f7',
    'reference-outputs.json': '7c3090a49c89f37c7bb68f23e4d1751a9f0e0a43d1e17753c624a061e8b6472e',
}


def validate(package, check_external=True):
    package = Path(package)
    directory = package / 'restoration/original-check'
    for name, expected in FIXTURES.items():
        if sha(directory / name) != expected:
            raise ValueError('exact restoration input absent/changed: ' + name)
    tokens = read(directory / 'input-8192.json')['tokens']
    if len(tokens) < 8192 or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('original 8192 input-token fixture invalid')
    for offset in (0, 1024, 2560, 4096):
        if len(tokens[offset:offset + 512]) != 512:
            raise ValueError('original recovery prompt slice incomplete')
    if len(read(directory / 'reference-outputs.json')['A']) < 5:
        raise ValueError('fixed-prefix reference incomplete')
    pins = read(package / 'source/restoration-external-reference-pins.json')
    reference = BASE.parent / 'step51-e7-service-20260930'
    required = {
        str(reference / f'P-final-greedy-d1-o{off}.json')
        for off in (0, 1024, 2560, 4096)
    }
    required.update(str(reference / f'P-final-{rep}-N1-o{off}.json')
                    for rep in range(3) for off in (0, 1024, 2560, 4096))
    required.update(str(reference / f'P-fixed-{mode}.json')
                    for mode in ('natural', 'full', 'mask'))
    required.add(str(reference / 'baseline-step51.json'))
    required.add(str(BASE.parent / 'step48-service-20260929/reference-state.json'))
    if set(pins) != required:
        raise ValueError('complete source-declared recovery references required')
    if check_external:
        for path, expected in pins.items():
            if sha(path) != expected:
                raise ValueError('original recovery external reference changed')
    return pins
