"""DEPRECATED: ralrc.model — import from ralrc.model_clean instead.

Original model.py used torch.cdist which lacks a registered _cdist_backward
kernel on several CUDA builds (torch 2.11+cu128 and later).  It also returned
a (E, F) tuple rather than the dict API used by train.py and eval.py.

This module re-exports ChargeAwarePotentialClean under the legacy
ChargeAwarePotential name so that any remaining import sites keep working.
The thin subclass accepts (and ignores) the legacy use_charge kwarg.

Do not add new code here.
"""
import warnings

from .model_clean import ChargeAwarePotentialClean

warnings.warn(
    "ralrc.model is deprecated; import ChargeAwarePotentialClean from "
    "ralrc.model_clean instead.  This shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)


class ChargeAwarePotential(ChargeAwarePotentialClean):
    """Legacy alias.  use_charge kwarg is accepted but ignored (use_coulomb controls both)."""

    def __init__(
        self,
        hidden: int = 64,
        n_elements: int = 119,
        use_charge: bool = True,
        use_coulomb: bool = True,
        **kwargs,
    ):
        super().__init__(hidden=hidden, n_elements=n_elements, use_coulomb=use_coulomb)
