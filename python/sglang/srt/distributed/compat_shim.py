# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sglang project
"""Minimal ``torch.distributed`` compatibility shim.

Some torch wheels do not ship a functional C++ distributed backend and therefore
leave ``torch.distributed`` as a stub: the module imports, but the runtime
classes/functions it exposes (``Backend``, ``ProcessGroup``, ``ReduceOp``,
``is_initialized``, ``new_group``, ...) are absent.  This happens on the ROCm
Windows wheels (tested: torch 2.11.0+rocm7.13.0), where the NCCL/RCCL extension
is not built and ``torch.distributed.is_available()`` returns ``False``.

sglang is designed to run serially on a single device when
``torch.distributed.is_available()`` is ``False`` (see the guards in
``scheduler.py``).  In that mode the collective code paths in
``parallel_state.py`` are never exercised, so the only requirement for the
single-GPU flow to import and run is that the *module-level* references resolve:
``from torch.distributed import Backend, ProcessGroup`` and
``int(torch.distributed.ReduceOp.SUM)``.

This module fills in exactly those missing symbols (and a handful of
collective stubs as defensive no-ops) so the rest of sglang imports cleanly.
The injected symbols are no-ops; they must never run for a real multi-rank job.
Detection is conservative: we only patch an attribute when it is already absent,
so a real torch.distributed keeps its native implementation.
"""

from __future__ import annotations

import enum
import torch.distributed as _td

_PATCHED = False


def patch_torch_distributed() -> bool:
    """Idempotently backfill missing ``torch.distributed`` symbols.

    Returns True if the shim installed any attributes (i.e. the wheel was a
    stub), False if torch.distributed was already complete (no patching
    needed).
    """
    global _PATCHED
    if _PATCHED:
        return False
    _PATCHED = True

    if getattr(_td, "_aicompass_shim", False):
        return False

    installed = False

    if not hasattr(_td, "is_available"):
        _td.is_available = lambda: False
        installed = True

    if not hasattr(_td, "is_initialized"):
        _td.is_initialized = lambda: False
        installed = True

    def _noop_group(_group=None, *args, **kwargs):
        return None

    def _noop(*args, **kwargs):
        return None

    def _world_size(_group=None):
        return 1

    def _rank(_group=None):
        return 0

    # Permissive stub class: any attribute access (including nested lookups
    # like ``ProcessGroup.BackendType.NCCL`` or ``Backend.UNDEFINED``) returns
    # a no-op callable instead of raising AttributeError.  Used for classes
    # that downstream code introspects (Backend, ProcessGroup) but that sglang
    # never actually invokes in single-device mode.
    class _AnyAttr:
        # intercept instance attribute access (e.g. x.something)
        def __getattr__(self, _name):
            return _noop
        def __call__(self, *a, **k):
            return None

    # Metaclass that intercepts *class-level* attribute access (e.g.
    # ``ProcessGroup.BackendType``) and returns an _AnyAttr instance, so the
    # nested member lookup (``ProcessGroup.BackendType.UNDEFINED``) then resolves
    # via instance __getattr__.  Without this, distributed_c10d.py line 307
    # (`ProcessGroup.BackendType`) raises AttributeError because the class
    # object itself has no such attribute.
    class _MetaAnyAttr(type):
        def __getattr__(cls, _name):
            return _AnyAttr()

    def _any_attr_cls(name):
        return _MetaAnyAttr(name, (_AnyAttr,), {})

    # Defensive no-op stubs for collectives. In single-device mode the sglang
    # guards (is_available()==False / world_size==1) never reach these, but
    # other import-time or debug code may reference them.
    _stubs = {
        "new_group": _noop_group,
        "init_process_group": _noop_group,
        "destroy_process_group": _noop_group,
        "barrier": _noop_group,
        "monitored_barrier": _noop_group,
        "all_reduce": _noop_group,
        "all_reduce_tensor": _noop,
        "reduce": _noop_group,
        "reduce_sum": _noop_group,
        "all_to_all_single": _noop,
        "all_to_all": _noop_group,
        "all_gather_into_tensor": _noop,
        "all_gather": _noop_group,
        "all_gather_object": _noop,
        "gather_object": _noop,
        "gather": _noop,
        "broadcast": _noop,
        "broadcast_object_list": _noop,
        "get_rank": _rank,
        "get_world_size": _world_size,
        "get_group_rank": _rank,
        "get_backend": _world_size,
        "isend": _noop,
        "irecv": _noop,
        "send": _noop,
        "recv": _noop,
        "Work": type("Work", (), {}),
        "Store": getattr(_td, "TCPStore", None) or type("Store", (), {}),
        "GroupMember": type("GroupMember", (), {}),
    }
    for _name, _fn in _stubs.items():
        if not hasattr(_td, _name):
            setattr(_td, _name, _fn)
            installed = True

    # torch/distributed/__init__.py only defines user-facing classes
    # (FileStore, Store, TCPStore, ProcessGroup, Work, Backend, Reducer, ...)
    # inside an `if is_available():` gate (is_available checks hasattr(torch._C,
    # "_c10d_init")).  On stub ROCm Windows wheels the C++ probe is absent, so
    # the gate is skipped and those names are never exported.  Yet
    # torch.distributed.distributed_c10d / _functional_collectives /
    # _symmetric_memory do `from torch.distributed import FileStore` at import
    # time, which crashes before the single-device path ever runs.
    #
    # Backfill exactly the names the gated block would have defined, as no-op
    # types/calls, so those imports resolve.  We do NOT override
    # is_available(): it must keep returning False so sglang's own guards
    # (parallel_state, scheduler) select the single-device flow instead of
    # attempting real collectives the stub cannot run.
    import enum as _enum

    if not hasattr(_td, "DebugLevel"):
        _td.DebugLevel = _enum.IntEnum(
            "DebugLevel", {"OFF": 0, "INFO": 1, "DETAIL": 2}
        )
        installed = True

    if not hasattr(_td, "ReduceOp"):
        # ReduceOp carries a nested RedOpType enum that distributed_c10d.py
        # introspects via `ReduceOp.RedOpType.__members__.items()` (deprecated
        # reduce_op alias).  Provide real IntEnum members so that resolves;
        # values are never dispatched in single-device mode.
        _RedOpType = _enum.IntEnum(  # type: ignore[assignment]
            "RedOpType",
            {"MIN": 0, "MAX": 1, "SUM": 2, "PRODUCT": 3,
             "BAND": 4, "BOR": 5, "BXOR": 6, "AVG": 7, "UNDEFINED": 8},
        )

        class _ReduceOp:
            RedOpType = _RedOpType
            SUM = _RedOpType.SUM
            PRODUCT = _RedOpType.PRODUCT
            MIN = _RedOpType.MIN
            MAX = _RedOpType.MAX
            BAND = _RedOpType.BAND
            BOR = _RedOpType.BOR
            BXOR = _RedOpType.BXOR
            AVG = _RedOpType.AVG
            UNDEFINED = _RedOpType.UNDEFINED

        _td.ReduceOp = _ReduceOp  # type: ignore[attr-defined]
        installed = True

    if not hasattr(_td, "reduce_op"):
        _td.reduce_op = getattr(_td, "ReduceOp")  # type: ignore[attr-defined]
        installed = True

    _td_types = (
        "FileStore", "PrefixStore", "Store", "TCPStore",
        "Reducer", "Logger", "GradBucket", "BuiltinCommHookType", "Work",
    )
    for _n in _td_types:
        if not hasattr(_td, _n):
            setattr(_td, _n, _any_attr_cls(_n))  # type: ignore[attr-set]
            installed = True

    # ProcessGroup/Backend are introspected by distributed_c10d
    # (ProcessGroup.BackendType.UNDEFINED, Backend.NCCL, ...) but never actually
    # dispatched in single-device mode.  Use the permissive _AnyAttr so nested
    # lookups resolve without enumerating every member.
    for _n in ("ProcessGroup", "Backend"):
        if not hasattr(_td, _n):
            setattr(_td, _n, _any_attr_cls(_n))
            installed = True

    _td_funcs = (
        "_broadcast_coalesced", "_compute_bucket_assignment_by_size",
        "_ControlCollectives", "_DEFAULT_FIRST_BUCKET_BYTES",
        "_make_nccl_premul_sum", "_register_builtin_comm_hook",
        "_register_comm_hook", "_StoreCollectives", "_test_python_store",
        "_verify_params_across_processes",
    )
    for _n in _td_funcs:
        if not hasattr(_td, _n):
            setattr(_td, _n, lambda *a, **k: None)
            installed = True

    for _setter in ("set_debug_level", "set_debug_level_from_env"):
        if not hasattr(_td, _setter):
            setattr(_td, _setter, lambda *a, **k: None)
            installed = True

    # ``torch.distributed.group`` is a submodule that some code path attributes
    # via ``dist.group.<x>``. Alias it to the module itself so attribute access
    # resolves to the patched module.
    if not hasattr(_td, "group"):
        _td.group = _td
        installed = True

    if not hasattr(_td, "TCPStore"):
        setattr(_td, "TCPStore", _any_attr_cls("TCPStore"))
        installed = True

    # ``StatelessProcessGroup`` in distributed/utils.py annotates a field as
    # ``store: torch._C._distributed_c10d.Store``.  On stub wheels that C++
    # extension module is absent, so the annotation raises AttributeError at
    # class-definition time.  Provide a no-op Store so the dataclass imports.
    #
    # ``torch._C`` is not a package (no ``__path__``), so ``import
    # torch._C._distributed_c10d`` (the form used internally by
    # ``torch.distributed.distributed_c10d``, ``_functional_collectives`` and
    # ``_symmetric_memory``) raises ``ModuleNotFoundError`` even if we set the
    # attribute on ``_C``.  Registering the shim module in ``sys.modules`` makes
    # those ``import`` statements resolve to the no-op module instead of walking
    # ``torch._C.__path__`` and failing.  Without this, every downstream
    # ``distributed_c10d``/``_functional_collectives``/``_symmetric_memory``
    # import crashes on a stub wheel.
    import torch._C as _C  # type: ignore[attr-defined]
    import types as _types
    import sys as _sys
    import enum as _enum

    # Build the no-op shim module once and reuse it across both branches.
    _needs_module = not hasattr(_C, "_distributed_c10d")

    def _make_c10d_stub() -> "_types.ModuleType":
        _c10d = _types.ModuleType("torch._C._distributed_c10d")

        # ``torch.distributed.distributed_c10d`` does a top-level
        # ``from torch._C._distributed_c10d import ( ...)`` (see
        # torch/distributed/distributed_c10d.py).  Backfill every name it
        # pulls so the import resolves on a stub wheel, instead of raising
        # ``ImportError: cannot import name '_DistributedBackendOptions'``.
        _type_names = {
            # --- classes/types ---
            "_DistributedBackendOptions",
            "AllgatherOptions",
            "AllreduceCoalescedOptions",
            "AllreduceOptions",
            "AllToAllOptions",
            "BarrierOptions",
            "BroadcastOptions",
            "GatherOptions",
            "PrefixStore",
             "ProcessGroup",
            "ReduceOptions",
            "ReduceScatterOptions",
            "ScatterOptions",
            "Store",
            "Work",
            # --- names only referenced by _symmetric_memory ---
            "_SymmetricMemory",
            "Backend",
        }
        for _n in _type_names:
            if not hasattr(_c10d, _n):
                # All class stubs use the permissive _MetaAnyAttr so class-level
                # introspection (e.g. ProcessGroup.BackendType, ReduceOp.RedOpType)
                # resolves without enumerating every nested member.
                setattr(_c10d, _n, _any_attr_cls(_n))  # type: ignore[arg-type]

        _RedOpType = _enum.IntEnum(  # type: ignore[assignment]
            "RedOpType",
            {"MIN": 0, "MAX": 1, "SUM": 2, "PRODUCT": 3,
             "BAND": 4, "BOR": 5, "BXOR": 6, "AVG": 7, "UNDEFINED": 8},
        )

        class _ReduceOp:
            RedOpType = _RedOpType
            SUM = _RedOpType.SUM
            PRODUCT = _RedOpType.PRODUCT
            MIN = _RedOpType.MIN
            MAX = _RedOpType.MAX
            BAND = _RedOpType.BAND
            BOR = _RedOpType.BOR
            BXOR = _RedOpType.BXOR
            AVG = _RedOpType.AVG
            UNDEFINED = _RedOpType.UNDEFINED

        if not hasattr(_c10d, "ReduceOp"):
            _c10d.ReduceOp = _ReduceOp  # type: ignore[attr-defined]
            installed = True

        # ``DebugLevel`` is an IntEnum in the real module and is compared by
        # value elsewhere; a plain IntEnum keeps those comparisons safe.
        if not hasattr(_c10d, "DebugLevel"):
            _DebugLevel = _enum.IntEnum("DebugLevel", {"OFF": 0, "INFO": 1, "DETAIL": 2})
            _c10d.DebugLevel = _DebugLevel  # type: ignore[attr-defined]

        # Callable no-ops for the remaining top-level references.
        for _fn in ("get_debug_level", "_register_process_group",
                    "_resolve_process_group", "_unregister_all_process_groups",
                    "_unregister_process_group", "_register_work",
                    "_is_nvshmem_available", "_set_global_rank",
                    "_hash_tensors"):
            if not hasattr(_c10d, _fn):
                setattr(_c10d, _fn, lambda *a, **k: None)  # type: ignore[attr-set]

        # Catch-all so any *new* name added by a future torch version also
        # resolves to a no-op instead of raising AttributeError on import.
        def __getattr__(name: str):  # noqa: ANN001
            return lambda *a, **k: None

        _c10d.__getattr__ = __getattr__  # type: ignore[method-assign]
        return _c10d

    if _needs_module:
        _c10d = _make_c10d_stub()
        _C._distributed_c10d = _c10d  # type: ignore[attr-defined]
        _sys.modules.setdefault("torch._C._distributed_c10d", _c10d)
        installed = True
    else:
        if not hasattr(_C._distributed_c10d, "Store"):
            # Re-point a real-but-incomplete extension at our catch-all so
            # missing symbols still resolve instead of raising ImportError.
            def _c10d_getattr(name: str):  # noqa: ANN001
                return getattr(_C._distributed_c10d, name, None) or (lambda *a, **k: None)
            _C._distributed_c10d.__getattr__ = _c10d_getattr  # type: ignore[attr-defined]
            installed = True

    _td._aicompass_shim = True  # type: ignore[attr-defined]
    return installed
