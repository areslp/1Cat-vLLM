"""Epoch-scoped adapter over the byte-identical controller ID contract."""
from request_ids import bind_scheduler_ids


class Bindings:
    def __init__(self): self.external_to_internal = {}

    def bind(self, internal, allowed):
        previous = {key: value for key, value in self.external_to_internal.items()
                    if key in allowed}
        updated = bind_scheduler_ids([internal], list(allowed), previous)
        self.external_to_internal.update(updated)
        return next(key for key, value in updated.items() if value == internal)

    def receipt(self, externals):
        return {external: self.external_to_internal[external]
                for external in externals if external in self.external_to_internal}
