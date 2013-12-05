# -*- coding: utf-8 -*-
from __future__ import print_function, division

__all__ = [
    "MathWarning",
    "EinsteinSummationWarning",
]

__submodules__ = [
    "abstract"
]

__all__ += __submodules__

class MathWarning(Warning):
    """
    A warning for when something potentially unexpected is going to happen mathematically.
    """

class EinsteinSummationWarning(MathWarning):
    """
    A warning for when something potentially unexpected is going to happen in an Einstein
    summation notation expression.
    """

class EinsteinSummationError(Exception):
    """
    All exceptions in this module will inherit from this class.
    """

class EinsteinSummationTemporaryObjectError(EinsteinSummationError):
    """
    An error involving a temporary object created (or errantly not created) by
    TensorInterfaceBase.get_temporary()
    """

class EinsteinSummationTemporaryObjectWarning(EinsteinSummationWarning):
    """
    An warning involving a temporary object created by
    TensorInterfaceBase.get_temporary()
    """

class EinsteinSummationIndexingError(EinsteinSummationError):
    """
    An error involving indexing or index parsing.
    """

class EinsteinSummationAlignmentError(EinsteinSummationError):
    """
    An error involving a shape mismatch
    """

type_checking_enabled = False
sanity_checking_enabled = False
