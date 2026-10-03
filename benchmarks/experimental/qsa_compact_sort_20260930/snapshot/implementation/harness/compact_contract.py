"""CPU ordering/extent bookkeeping only, not a replacement for CUDA evidence."""

CAPACITY = 8192
THREADS = 512
EMPTY = 0xFFFFFFFF
LOW36 = (1 << 36) - 1


def category(mask):
    tiles = 0
    for query in range(8):
        if mask & (15 << (query * 4)):
            tiles |= 1 << (query * 6 // 16)
            tiles |= 1 << ((query * 6 + 5) // 16)
    return tiles


def bucket(count):
    if not 0 <= count <= CAPACITY:
        raise ValueError("hash capacity exceeded")
    return next((size for size in (1024, 2048, 4096, 8192) if count <= size))


def stable_compact(slots, owners):
    """Model blocked register capture followed by thread-count exclusive scan."""
    if len(slots) != CAPACITY or len(owners) != CAPACITY:
        raise ValueError("wrong hash geometry")
    registers = list(zip(slots, owners))
    counts = [sum(entry & EMPTY != EMPTY for entry, _ in
                  registers[tid * 16:(tid + 1) * 16]) for tid in range(THREADS)]
    output = [None] * sum(counts)
    offset = 0
    for tid, count in enumerate(counts):
        cursor = offset
        for entry, owner in registers[tid * 16:(tid + 1) * 16]:
            if entry & EMPTY != EMPTY:
                output[cursor] = (entry, owner)
                cursor += 1
        assert cursor == offset + count
        offset += count
    return output


def ordered(entries):
    def key(pair):
        entry, owner = pair
        return (((category(entry >> 32) << 32) | owner)
                if entry & EMPTY != EMPTY else (1 << 64) - 1) & LOW36
    return sorted(entries, key=key)


def scatter(sorted_entries, width=4160):
    pages, masks = [-777] * width, [0xA5A5A5A5] * width
    groups = [[] for _ in range(8)]
    for entry, _ in sorted_entries:
        if entry & EMPTY != EMPTY:
            groups[category(entry >> 32)].append(entry)
    offset, first = 0, next((entry & EMPTY for group in groups
                            for entry in group), None)
    for group in groups[1:]:
        padded = (len(group) + 7) & ~7
        for rank in range(padded):
            index = offset + rank
            if index < width:
                pages[index] = group[rank] & EMPTY if rank < len(group) else first
                masks[index] = group[rank] >> 32 if rank < len(group) else 0
        offset += padded
    return pages, masks, min(offset, width) * 4
