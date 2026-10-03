"""Private graph-resident comparison. Imported only in shadow mode.

No float arithmetic/comparison tolerance; float16/float32 compared via raw bits.
No per-node CPU transfer, .item(), or synchronization. Preparation is separate.
"""


class Buffer:
    GUARD = 64

    def __init__(self, torch, shape, dtype, device, sentinel):
        self.torch, self.shape, self.sentinel = torch, tuple(shape), sentinel
        count = 1
        for size in shape:
            count *= size
        self.raw = torch.empty(count + 2 * self.GUARD, dtype=dtype, device=device)
        self.value = self.raw[self.GUARD:-self.GUARD].view(shape)
        self.reset()

    def reset(self):
        if self.raw.dtype == self.torch.float16:
            self.raw.view(self.torch.int16).fill_(self.sentinel)
        elif self.raw.dtype == self.torch.float32:
            self.raw.view(self.torch.int32).fill_(self.sentinel)
        else:
            self.raw.fill_(self.sentinel)

    def guard_bad(self):
        bits = self.raw
        if bits.dtype == self.torch.float16:
            bits = bits.view(self.torch.int16)
        elif bits.dtype == self.torch.float32:
            bits = bits.view(self.torch.int32)
        return ((bits[:self.GUARD] != self.sentinel).any()
                | (bits[-self.GUARD:] != self.sentinel).any())

    def nbytes(self):
        return self.raw.numel() * self.raw.element_size()


class Shadow:
    def __init__(self, torch, q):
        self.torch = torch
        groups = q.shape[0] // 8
        self.pages = Buffer(torch, (groups, 4160), torch.int32, q.device, -777)
        self.masks = Buffer(torch, (groups, 4160), torch.uint32,
                            q.device, 0xA5A5A5A5)
        self.lengths = Buffer(torch, (groups,), torch.int32, q.device, -777)
        self.out = Buffer(torch, q.shape, q.dtype, q.device, 0x7E55)
        self.lse = Buffer(torch, q.shape[:2], torch.float32,
                          q.device, 0x7FC12345)
        self.buffers = (self.pages, self.masks, self.lengths, self.out, self.lse)
        self.columns = torch.arange(4160, device=q.device)[None, :]
        self.bytes = sum(buf.nbytes() for buf in self.buffers)
        self.bytes += self.columns.numel() * self.columns.element_size()
        self.layout = {'q_shape': list(q.shape), 'q_dtype': str(q.dtype),
                       'addresses': [b.raw.data_ptr() for b in self.buffers],
                       'private_bytes': self.bytes}

    def reset(self):
        for buf in self.buffers:
            buf.reset()

    def workspace(self):
        return (self.pages.value, self.masks.value, self.lengths.value,
                self.lse.value)

    def compare(self, original_workspace, original_out, counter):
        torch = self.torch
        rp, rm, rl, r_lse = original_workspace
        cp, cm, cl, c_lse = self.workspace()
        lengths_bad = (rl != cl).any() | (cl < 0).any() | (cl > 4160 * 4).any() | (cl % 4 != 0).any()
        valid = self.columns < rl[:, None] // 4
        masks_different = (rm.view(torch.int32) != cm.view(torch.int32))
        meta_bad = lengths_bad | (((rp != cp) | masks_different) & valid).any()
        private_valid = self.columns < cl[:, None] // 4
        tails_bad = (((cp != -777) | (cm.view(torch.int32) != -1515870811))
                     & ~private_valid).any()
        out_bad = (original_out.contiguous().view(torch.int16)
                   != self.out.value.view(torch.int16)).any()
        lse_bad = (r_lse.contiguous().view(torch.int32)
                   != c_lse.view(torch.int32)).any()
        guard_bad = self.pages.guard_bad()
        for buf in self.buffers[1:]:
            guard_bad = guard_bad | buf.guard_bad()
        # This add is recorded after the exact candidate planner/forward node.
        counter[0].add_(1)
        for idx, bad in enumerate((meta_bad, out_bad, lse_bad, guard_bad,
                                   tails_bad), 1):
            counter[idx].add_(bad.to(torch.int64))
        return meta_bad | out_bad | lse_bad | guard_bad | tails_bad
