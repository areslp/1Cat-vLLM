"""Shadow-bank graph smoke: synthetic only, no model/library build. Opt-in window."""
import argparse
import os
import resource
from pathlib import Path
import torch
from first_failure import FirstFailure, values
from shadow import Shadow
from io_contract import save

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--device', type=int, default=0)
a = p.parse_args()
torch.set_num_threads(1); torch.set_num_interop_threads(1)
assert a.device == 0 and torch.cuda.device_count() == 1
assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
device = torch.device('cuda', a.device); torch.cuda.set_device(device)
q = torch.zeros((40, 6, 256), dtype=torch.float16, device=device)
q.view(torch.int16).reshape(-1)[:4].copy_(torch.tensor(
    [0, -32768, 0x7E55, 0x7F31], dtype=torch.int16, device=device))
private = Shadow(torch, q)
reference = (torch.zeros((5, 4160), dtype=torch.int32, device=device),
             torch.zeros((5, 4160), dtype=torch.uint32, device=device),
             torch.full((5,), 32, dtype=torch.int32, device=device),
             torch.zeros((40, 6), dtype=torch.float32, device=device))
arguments = {'q': q, 'logical_indices': torch.zeros((40, 416), dtype=torch.int32, device=device),
             'block_table': torch.zeros((8, 64), dtype=torch.int32, device=device),
             'token_to_req': torch.arange(40, dtype=torch.int32, device=device) // 5,
             'query_positions': torch.arange(40, dtype=torch.int64, device=device),
             'sequence_lengths': torch.ones(8, dtype=torch.int32, device=device),
             'out': q.clone()}
example = values(arguments, reference, private)
bank = FirstFailure(torch, example, 64 * 1024**2); bank.prepare(0, example)
assert all(bank.bank[k].untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
           for k, v in example.items())
trigger = torch.zeros((), dtype=torch.bool, device=device)
counter = torch.zeros(6, dtype=torch.int64, device=device)
def comparison():
    private.reset(); private.lengths.value.fill_(32)
    private.pages.value[:, :8].zero_(); private.masks.value.view(torch.int32)[:, :8].zero_()
    private.out.value.copy_(q); private.lse.value.zero_()
    value = private.out.value.view(torch.int16).reshape(-1)[:1]
    value.copy_(torch.where(trigger, value ^ 1, value))
    bad = private.compare(reference, q, counter)
    bank.capture(0, values(arguments, reference, private), bad)
stream = torch.cuda.Stream(device=device); stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream): comparison()
torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
bank.reset(); counter.zero_()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph): comparison()
bank.reset(); counter.zero_(); trigger.fill_(False); graph.replay(); torch.cuda.synchronize()
assert bank.flag.cpu().tolist() == [0, 0, 0, 0]
original = q.view(torch.int16).cpu().clone()
trigger.fill_(True); graph.replay(); torch.cuda.synchronize()
first = bank.bank['q'].view(torch.int16).cpu().clone()
assert torch.equal(first, original)
q.view(torch.int16).fill_(0x7EAA); graph.replay(); torch.cuda.synchronize()
assert torch.equal(bank.bank['q'].view(torch.int16).cpu(), first)
output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
pt = output.with_suffix('.pt'); bank.export(pt, 64 * 1024**2, {'synthetic_smoke': True})
loaded = torch.load(pt, map_location='cpu', weights_only=True)
assert torch.equal(loaded['values']['q'].view(torch.int16), first)
bank.reset(); graph.replay(); torch.cuda.synchronize()
assert torch.equal(bank.bank['q'].view(torch.int16).cpu(), q.view(torch.int16).cpu())
assert counter.cpu().tolist()[1:] == [0, 3, 0, 0, 0]
assert torch.cuda.max_memory_allocated(device) <= 64 * 1024**2
assert torch.cuda.max_memory_reserved(device) <= 128 * 1024**2
save(output, {'status': 'PASS_SYNTHETIC_SHADOW_GRAPH_NOT_SERVICE',
              'bank_bytes': bank.bytes, 'private_bytes': private.bytes,
              'cpu_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'allocated_peak': torch.cuda.max_memory_allocated(device),
              'reserved_peak': torch.cuda.max_memory_reserved(device),
              'device': str(device), 'torch': torch.__version__,
              'python': __import__('sys').version,
              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
              'device_uuid': str(torch.cuda.get_device_properties(device).uuid),
              'checks': ['false', 'first_fail', 'second_fail_immutable',
                         'safe_drain', 'reset_next_epoch', 'guards', 'disjoint_storage'],
              'not_proved': ['real candidate/service graph', 'KV attention replay']})
