"""Fixed bijection between legal combos and ComboTable cells.

Every legal Regicide combo occupies exactly one ``(location, played_status)``
cell of the ComboTable action space (see ``core/combotable.cc`` ``setComboEntry``),
and the location bitmask :pyattr:`regi_py.core.Combo.bitwise` is a unique identity
for the combo's card set.  This module precomputes the two-way mapping so callers
can convert a combo's ``bitwise`` to/from its ``(loc, pst)`` cell without building
a throwaway ``ComboTable`` per lookup.

The mapping is a genuine bijection: enumerating every viable cell (via
``ComboTable.all_entries``) yields distinct ``bitwise`` values, one per cell.

**Yield is special.**  The yield combo is the *empty* combo (``bitwise == 0``),
but its cell is ``(0, PLAYED_SELF)`` and ``make_combo(0, 0)`` carries the yield
*card* (``bitwise == 1``).  Both directions here normalize yield to ``bitwise 0``
-- the identity a played yield actually has -- so ``1`` is never a key.
"""
import numpy as np

from .core import ComboTable

__all__ = [
    "cell_of_bitwise",
    "bitwise_of_cell",
    "bitwise_to_cell_map",
]

# (LOCATION_YIELD, PLAYED_SELF)
_YIELD_CELL = (0, 0)

_bitwise_to_cell = None


def _build():
    table = ComboTable.all_entries()
    arr = np.asarray(table)
    mapping = {}
    for loc, pst in zip(*arr.nonzero()):
        loc, pst = int(loc), int(pst)
        if (loc, pst) == _YIELD_CELL:
            mapping[0] = _YIELD_CELL  # played yield is the empty combo (bitwise 0)
            continue
        mapping[ComboTable.make_combo(loc, pst).bitwise] = (loc, pst)
    return mapping


def bitwise_to_cell_map():
    """The full ``{bitwise: (loc, pst)}`` mapping, built once and cached."""
    global _bitwise_to_cell
    if _bitwise_to_cell is None:
        _bitwise_to_cell = _build()
    return _bitwise_to_cell


def cell_of_bitwise(bitwise, default=None):
    """Return the ``(loc, pst)`` cell for a combo's ``bitwise``, else ``default``."""
    return bitwise_to_cell_map().get(bitwise, default)


def bitwise_of_cell(loc, pst):
    """Return the ``bitwise`` identity of the combo at cell ``(loc, pst)``.

    The reverse of :func:`cell_of_bitwise`; the yield cell normalizes to ``0``.
    Returns ``0`` for any cell whose ``make_combo`` is empty (invalid cell).
    """
    if (loc, pst) == _YIELD_CELL:
        return 0
    return ComboTable.make_combo(loc, pst).bitwise
