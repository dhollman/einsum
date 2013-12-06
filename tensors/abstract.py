# -*- coding: utf-8 -*-
"""
Abstract classes and such which tensor interfaces should inherit from.
"""
from __future__ import print_function, division
from abc import ABCMeta, abstractmethod, abstractproperty

class TensorInterfaceBase(object):
    """
    The abstract base class for the tensor interface concept that
    the python Einstein summation layer interfaces with.  Tensor
    interfaces must inherit from this class.
    """
    __metaclass__ = ABCMeta

    #--------------------------------------------------------------------------------#

    #region | Abstract Class Methods                                                    {{{1 |

    # Note:  Python 2.x does not have the @abstractclassmethod that Python 3 has.
    #   Just make sure you implement these methods.  There is no safety net if you
    #   don't.

    @classmethod
    def create_tensor(cls, shape):
        """
        Should return a temporary of type `cls` with shape given by `shape`.  The `shape`
        parameter will be a `tuple` of `int`.
        """
        return NotImplemented

    @classmethod
    def release_tensor(cls, obj):
        """
        Called to notify the underlying implementation that the tensor `obj` is no
        longer needed.  Note that `obj` must have been created using `cls.create_tensor()`;
        if not, this method should raise an `EinsteinSummationError`.  (Thus,
        the implementation should keep track of the tensors given out by `create_tensor()`).
        Good implementations should also check for unreleased tensors at exit using
        the Python standard library `atexit` module and raise a warning.
        """
        return NotImplemented

    @classmethod
    def dot_product(cls, alpha, *args):
        """
        Return the dot product of the TensorInterfaceBase instances given in args using
        indices given in args.  Arguments are given as tensor1, indices1, tensor2, indices2, etc.
        This may be simplified with a default implementation that calls some simpler (pairwise)
        dot product/contract function, creating temporaries along the way.  Even so, it's
        probably more efficient to hook a subclass into this method and handle dot product
        factoring in the interface.
        """
        return NotImplemented


    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Abstract Properties                                                       {{{1 |

    @abstractproperty
    def shape(self):
        """
        The shape of the underlying tensor as a tuple of ints.  For instance,
        if a tensor is 4x6x4x6, this should return (4, 6, 4, 6).  Vectors
        (i.e. one dimensional tensors) should return a tuple with one element.
        """
        return NotImplemented

    #endregion }}}1
    #--------------------------------------------------------------------------------#


    #region | Abstract Methods                                                          {{{1 |

    @abstractmethod
    def contract_into(self, alpha, a, a_indices, b, b_indices, beta, c_indices):
        """
        Perform the Einstein summation contraction
            self[c_indices] = alpha * a[a_indices] * b[b_indices] + beta * self[c_indices]
        Returns `None`.  The contraction should actually be carried out at the time
        of the call, or at least securely wrapped in an underlying future managed
        by the implementation.
        """
        return NotImplemented

    @abstractmethod
    def sort_into(self, old, new_axes):
        """
        Sort `old` into `self`.  The parameter `new_axes` will be a `tuple` of `int`
        specifying the new axis ordering.  For instance, if the user wanted to perform
        the sort
            self['i,a,j,b'] = old['i,j,a,b'],
        the `new_axes` parameter would be [0, 2, 1, 3].  Similiarly,
            self['d,a,b,c'] = old['a,b,c,d']
        would have a `new_axes` parameter of [3, 0, 1, 2].
        Returns `None`.
        """
        return NotImplemented

    @abstractmethod
    def add_into(self, alpha, a, beta):
        """
        Performs the Einstein summation expression
            self[...] = alpha * a[...] + beta * self[...]
        where "..." must be the same for `a` and `self`.  The implementation may
        assume that `self` and `a` have exactly the same shape *or* that `a` is
        exactly the integer 1 (i.e. `a is 1` returns True; note that `1.0` is
        not the same in this case), in which case the method is expected to add the
        constant alpha to all elements in `a` after scaling `self` by `beta`.
        Note that `a` can be the same as `self` (or a trivial view of `self`),
        and thus the following edge cases, for instance, should be handled with
        care:

        * t.add_into(1.0, t, 2.0)
        * t.add_into(1.0, t, 0.0)
        * t.add_into(1.0, t.subtensor_view(*[slice(i) for i in t.shape]), 0.0)
        """
        return NotImplemented

    @abstractmethod
    def subtensor_view(self, *args):
        """
        Returns a view of the tensor containing slices along the various axes.
        Note that the returned value should be a strict view, in the sense that
        modification of an entry in the returned value should modify the corresponding
        entry in `self`.  The returned value should be an instance of TensorInterfaceBase.
        Interfaces that do not support strides should return `NotImplemented` for slices
        with strides other than 1.  Note that `len(args)` will always be the same as
        `len(self.shape)`
        """
        return NotImplemented

    @abstractmethod
    def get_element(self, indices):
        """
        Returns the element in `self` described by `indices`.  The parameter `indices` must
        be a `tuple` of `int`.
        """
        return NotImplemented

    @abstractmethod
    def set_element(self, indices, value):
        """
        Sets the element in `self` described by `indices` to `value`.  The parameter `indices`
        must be a `tuple` of `int`, and `value` should be castable to the data type of
        elements in `self`.  Should return `None`.
        """
        return NotImplemented

    @abstractmethod
    def internal_dot(self, alpha, indices):
        """
        Evaluate expressions that are of the form e.g.
            T["i,j,i,j"]
        Should return a value that is the same type as the elements in `self`
        """
        return NotImplemented



    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Methods                                                                   {{{1 |

    # Note:  These methods include default implementations that call one or more
    #  of the required abstract methods above.  In some cases, there may be
    #  a more efficient way to do what these methods are doing in a given interface,
    #  in which case the interface should overload said method.

    def scale(self, alpha):
        """
        Scale the underlying tensor by the factor `alpha`.  This will almost
        always be more efficiently done by the interface in an overloaded version,
        but it can be done with what we have if not.  The default implementation
        calls `add_into()`.
        """
        self.add_into(alpha, self, 0.0)

    def copy_into(self, other):
        """
        Copies `other` into `self`.  The interface may assume that `other` and `self`
        have exactly the same shape.  Default implementation uses `add_into()` with
        a `beta` parameter of 0.
        """
        self.add_into(1.0, other, 0.0)

    def shares_data_with(self, other):
        """
        Return `True` if `self` is a view of `other` or `other` is a view of `self`.
        This is needed for expressions like
            T["i,j,k"] = T["k,i,j"] + T["j,k,i"]
        in which case an extra temporary needs to be allocated for the right-hand side.
        However, most of the time you don't want to allocate such a temporary; if data
        on the right hand side doesn't depend on the left hand side, the in-place solution
        is superior.
        At the cost of efficiency, an interface can simply return `True` for this method always,
        as the default implementation does, and copies will be made every time.  In other
        words, no part of the library depends on tensors that do not share data always
        returning `False` from this method.  When tensors do share data, however,
        undefined behavior may result from returning `False`.
        """
        return True

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Special Methods                                                           {{{1 |

    def __getitem__(self, item):
        """
        Particularly useful for testing; in general should not be used or overloaded
        by the interface.  Just calls `self.get_element()`
        """
        return self.get_element(item)

    def __setitem__(self, key, value):
        """
        Particularly useful for testing; in general should not be used or overloaded
        by the interface.  Just calls `self.set_element()`
        """
        return self.set_element(key, value)

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    # end of TensorInterfaceBase class

