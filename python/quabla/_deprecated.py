"""Deprecated public names, resolved lazily so that only explicit use warns.

Each name stays in `quabla.__all__` through 1.x, as the SemVer policy in
CONTRIBUTING.md requires, and is removed in 2.0.
"""

import dis

# Deprecated name -> what to use instead, named in the warning.
REPLACEMENTS = {
    "TensorCudaAdamOptimizer": 'qb.optim.Trainer(..., device="cuda")',
    "TensorMlxAdamOptimizer": 'qb.optim.Trainer(..., device="mlx")',
}


def is_silent_lookup(frame):
    """Whether the module `__getattr__` called from `frame` must not warn.

    Importing from a package reaches `__getattr__` from importlib's
    `_handle_fromlist`, which probes each requested name (each name of
    `__all__` for a star import) before the importing frame binds it. The
    probe never warns: `from quabla import name` then warns once from the
    importer's IMPORT_FROM, and a star import binds silently, so only
    explicit use warns. The importer sits on IMPORT_STAR (3.10, 3.11) or
    on CALL_INTRINSIC_1 with INTRINSIC_IMPORT_STAR (3.12+) during a star
    import.
    """
    if frame.f_code.co_name == "_handle_fromlist" and frame.f_globals.get(
        "__name__"
    ) in ("importlib._bootstrap", "_frozen_importlib"):
        return True
    for instruction in dis.get_instructions(frame.f_code):
        if instruction.offset == frame.f_lasti:
            return (
                instruction.opname == "IMPORT_STAR"
                or instruction.argrepr == "INTRINSIC_IMPORT_STAR"
            )
    return False


def message(name):
    return (
        f"quabla.{name} is deprecated and will be removed in 2.0; "
        f"use {REPLACEMENTS[name]}, which creates it."
    )
