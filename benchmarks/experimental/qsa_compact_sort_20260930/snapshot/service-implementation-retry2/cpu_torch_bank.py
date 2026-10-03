"""Bounded realTorch CPU-only synthetic bank/domain check; no model/CUDA init."""
import argparse
import json
from pathlib import Path
import tempfile
import torch
from first_failure import FirstFailure, values
from shadow import Shadow
from io_contract import save

p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
a = p.parse_args()
torch.set_num_threads(1); torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()
q = torch.zeros((40, 6, 256), dtype=torch.float16)
q.view(torch.int16).reshape(-1)[:4].copy_(torch.tensor(
    [0, -32768, 0x7E55, 0x7F31], dtype=torch.int16))
private = Shadow(torch, q)
reference = (torch.zeros((5, 4160), dtype=torch.int32),
             torch.zeros((5, 4160), dtype=torch.uint32),
             torch.full((5,), 32, dtype=torch.int32),
             torch.zeros((40, 6), dtype=torch.float32))
arguments = {'q': q, 'logical_indices': torch.zeros((40, 416), dtype=torch.int32),
             'block_table': torch.zeros((8, 64), dtype=torch.int32),
             'token_to_req': torch.arange(40, dtype=torch.int32) // 5,
             'query_positions': torch.arange(40, dtype=torch.int64),
             'sequence_lengths': torch.ones(8, dtype=torch.int32), 'out': q.clone()}
example = values(arguments, reference, private)
source = {k: v.clone() for k, v in example.items()}
bank = FirstFailure(torch, example, 64 * 1024**2); bank.prepare(0, example)
assert all(bank.bank[k].untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
           for k, v in example.items())
bank.capture(0, example, torch.tensor(False))
assert bank.flag.tolist() == [0, 0, 0, 0]
bank.capture(0, example, torch.tensor(True))
assert torch.equal(bank.bank['q'].view(torch.int16), q.view(torch.int16))
q.view(torch.int16).fill_(0x7EAA)
bank.prepare(1, example); bank.capture(1, example, torch.tensor(True))
assert torch.equal(bank.bank['q'].view(torch.int16), source['q'].view(torch.int16))
assert bank.flag.tolist()[1] == 0
out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
snapshot_dir = out.parent / 'synthetic-bank-attempt1'
snapshot_dir.mkdir(exist_ok=False)
path = snapshot_dir / 'failure.pt'
bank.export(path, 64 * 1024**2, {'cpu_only': True})
payload = torch.load(path, weights_only=True)
assert torch.equal(payload['values']['q'].view(torch.int16), source['q'].view(torch.int16))
old = path.read_bytes()
try: bank.export(path, 64 * 1024**2, {})
except FileExistsError: pass
else: raise AssertionError('first snapshot overwrite allowed')
assert path.read_bytes() == old
bank.reset(); bank.capture(1, example, torch.tensor(True))
assert bank.flag.tolist()[1] == 1
assert torch.equal(bank.bank['q'].view(torch.int16), q.view(torch.int16))
# Metadata valid prefix is tokens//4; raw original suffix is allowed stale.
private.reset(); private.lengths.value.fill_(32)
private.pages.value[:, :8].zero_(); private.masks.value.view(torch.int32)[:, :8].zero_()
reference[0][:, 8:].fill_(123456)
private.out.value.copy_(q); private.lse.value.zero_()
counter = torch.zeros(6, dtype=torch.int64)
bad = private.compare(reference, q, counter)
assert not bool(bad) and counter.tolist() == [1, 0, 0, 0, 0, 0]
assert not torch.cuda.is_initialized()
save(out, {'status': 'PASS_REAL_TORCH_CPU_BANK_NOT_CUDA',
           'torch': torch.__version__, 'python': __import__('sys').version,
           'cuda_initialized': torch.cuda.is_initialized(),
           'bank_bytes': bank.bytes, 'private_bytes': private.bytes,
           'checks': ['bitpayload', 'false_predicate', 'first_fail_immutable',
                      'exclusive_export', 'drain_load_weights_only', 'epoch_reset',
                      'input_bank_disjoint', 'tokens_pages_stale_suffix']})
