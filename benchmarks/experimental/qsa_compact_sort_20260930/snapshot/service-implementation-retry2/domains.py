"""Planner output seq lengths count TOKENS, four per physical page."""

def pages_for_tokens(value):
    assert type(value) is int and 0 <= value <= 4160 * 4 and value % 4 == 0
    return value // 4
