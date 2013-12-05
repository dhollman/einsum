# -*- coding: utf-8 -*-
"""
Tensor front end
"""
from __future__ import division, print_function
from tensors import EinsteinSummationIndexingError, EinsteinSummationAlignmentError
from tensors.indices import split_indices, IndexRange


class Tensor(object):
    """
    Wraps the implementation details of a given interface into a nice package
    """

    #--------------------------------------------------------------------------------#

    #region | Class Attributes                                                          {{{1 |

    interface = None
    """
    The name of the TensorImplementationBase subclass that this class should use by default.
    """

    #--------------------------------------------------------------------------------#

    #region | Initialization                                                            {{{1 |

    def __init__(self,
            indices,
            index_range_set=None,
            _impl=None,
            interface = None
    ):
        """
        TODO Thorough documentation
        """
        self.indices = split_indices(indices)
        if index_range_set is None:
            self.index_range_set = IndexRange.global_index_range_set
        else:
            self.index_range_set = index_range_set
        self.ranges = tuple(self.index_range_set[i] for i in self.indices)
        #----------------------------------------#
        if interface is None:
            interface = type(self).interface
        self._interface = interface
        #----------------------------------------#
        if _impl is not None:
            self._impl = _impl
            shape = []
            for r in self.ranges:
                try:
                    size = r.size
                    shape.append(size)
                except EinsteinSummationIndexingError:
                    shape.append(None)
            if len(shape) != len(_impl.shape):
                raise EinsteinSummationAlignmentError("shape mismatch")
            for my_size, expected in zip(shape, _impl.shape):
                if my_size is not None and my_size != expected:
                    raise EinsteinSummationAlignmentError("shape mismatch")
        #- - - - - - - - - - - - - - - - - - - - #
        else:
            shape = tuple(r.size for r in self.ranges)
            self._impl = interface.create_tensor(shape)
        #----------------------------------------#

    def __getitem__(self, item):
        return EinsumTensor(
            indices=item,
            tensor=self,
            coeff=1.0,
            index_range_set=self.index_range_set
        )

    def __setitem__(self, key, value):
        lhs = EinsumTensor(
            indices=key,
            tensor=self,
            coeff=1.0,
            index_range_set=self.index_range_set
        )
        # Note that the "_into" in these methods
        #   mean the backwards thing what they do in _impl...
        if isinstance(value, EinsumContraction):
            value.contract(lhs)
        elif isinstance(value, EinsumSum):
            value.sum_into(lhs)
        elif isinstance(value, EinsumTensor):
            value.sort_into(lhs)





    pass

from einsum import EinsumTensor, EinsumContraction, EinsumSum
