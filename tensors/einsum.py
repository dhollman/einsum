# -*- coding: utf-8 -*-
"""
Wrapper classes that are returned by Tensor.__getitem__(), etc.
"""
from __future__ import print_function, division
from operator import mul
from tensors import sanity_checking_enabled, EinsteinSummationAlignmentError
from tensors.indices import split_indices, IndexRange
from copy import copy
import string
from numbers import Real

# A set of indices that are safe to pass to TensorInterfaceBase instances
_index_set = string.ascii_lowercase

#TODO raise reasonable exceptions when interface sends back NotImplemented

class EinsumTensor(object):
    """
    Immutable wrapper class that is returned by Tensor.__getitem__()
    """

    #--------------------------------------------------------------------------------#

    #region | Attributes                                                                {{{1 |

    indices = None
    """
    The indices that the EinsumTensor was constructed with, as a `tuple` of (unicode) strings
    """

    ranges = None
    """
    The index ranges corresponding to `indices`
    """

    coeff = None
    """
    The coefficient of the EinsumTensor in the expression
    """

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Initialization                                                            {{{1 |

    def __new__(cls, indices, tensor, coeff, index_range_set, known_indices=True):
        """
        Construct an EinsumTensor with `indices` and a reference to `tensor`.
        This should never be called directly by the user; it is called by the
        __getitem__() method of Tensor.  Needs to be __new__() rather than __init__()
        to handle internal contractions of dot products
        """
        self = object.__new__(cls)
        #----------------------------------------#
        self.coeff = coeff
        self.indices = split_indices(indices)
        self._tensor = tensor
        self.ranges = []
        #----------------------------------------#
        if known_indices:
            self.known_indices = known_indices
            if index_range_set is None:
                index_range_set = IndexRange.global_index_range_set
            for idx in indices:
                if idx in index_range_set:
                    self.ranges.append(index_range_set[idx])
            self.ranges = tuple(self.ranges)
        else:
            raise NotImplementedError("EinsumTensor objects without declared"
                                      " ranges not implemented")
        #----------------------------------------#
        # See if it's fully internally contracted.  If so, perform the internal contraction
        #   and return a float
        if all(self.indices.count(i) == 2 for i in self.indices):
            idxs = list(set(self.indices))
            idxmap = {}
            for n, i in enumerate(idxs):
                idxmap[i] = _index_set[n]
            return self.sliced_tensor._impl.internal_dot(1.0,
                tuple(idxmap[i] for i in self.indices)
            )
        #----------------------------------------#
        # otherwise, return the EinsumTensor object
        return self

    #--------------------------------------------------------------------------------#

    #region | Properties                                                                {{{1 |

    @property
    def sliced_tensor(self):
        return Tensor(
            self.indices,
            index_range_set=self.index_range_set,
            _impl=self._tensor._impl.subtensor_view(*self.slices_in_parent),
            interface=self._tensor._interface()
        )

    @property
    def slices(self):
        return [r.slice for r in self.ranges]

    @property
    def slices_in_parent(self):
        return [r.slice_in(pr) for r, pr in zip(self.ranges, self._tensor.ranges)]

    @property
    def name(self):
        if hasattr(self._tensor, "name") and self._tensor.name is not None:
            return self._tensor.name
        else:
            return "(unnamed tensor)"

    #--------------------------------------------------------------------------------#

    #region | Special Methods                                                           {{{1 |

    def __copy__(self):
        return EinsumTensor(
            indices=self.indices,
            tensor=self._tensor,
            coeff=self.coeff,
            index_range_set=self.index_range_set,
            known_indices=self.known_indices
        )

    #----------------------#
    # Arithmetic Operators #
    #----------------------#

    def __neg__(self):
        rv = copy(self)
        rv.coeff *= -1.0
        return rv

    def __mul__(self, other):
        if isinstance(other, EinsumTensor):
            return EinsumContraction(self, other)
        elif isinstance(other, EinsumContraction):
            result = other.append_tensor(self)
            if result is not None:
                return result
            else:
                return other
        elif isinstance(other, Real):
            rv = copy(self)
            rv.coeff *= other
            return rv
        else: # pragma: no cover
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Real):
            return self.__mul__(other)
        else: # pragma: no cover
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, (EinsumTensor, EinsumContraction)):
            return EinsumSum(self, other)
        elif isinstance(other, EinsumSum):
            other.summands.append(self)
            return other
        else: # pragma: no cover
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (EinsumTensor, EinsumContraction, EinsumSum)):
            return EinsumSum(self, -other)
        else: # pragma: no cover
            return NotImplemented

    def __iadd__(self, other):
        # TODO write this; be carefull about overwriting data in contract() and sum_into()
        return NotImplemented

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Methods                                                                   {{{1 |

    def sort_into(self, other):
        """
        sort this and store the result in other; e.g.
            other['pqrs'] = self['qrsp']
        """
        new_axes = []
        for idx in self.indices:
            if idx not in other.indices:
                raise EinsteinSummationAlignmentError("mismatched lhs/rhs"
                                                      " indices: ({}) != ({})".format(
                    ", ".join(self.indices),
                    ", ".join(other.indices)
                ))
            new_axes.append(other.indices.index(idx))
        other.sliced_tensor._impl.sort_into(self.sliced_tensor._impl, new_axes)

    #--------------------------------------------------------------------------------#

    pass
    # end class EinsumTensor

class EinsumContraction(object):
    """
    A contraction between two EinsumTensors, an EinsumTensor and
    an EinsumContraction, or two EinsumContractions.
    """

    #--------------------------------------------------------------------------------#

    #region | Class Attributes                                                          {{{1 |

    print_factorization = False

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Attributes                                                                {{{1 |

    tensors = None

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Initialization                                                            {{{1 |

    def __new__(cls, *args):
        ret_val = object.__new__(cls)
        #----------------------------------------#
        ret_val.tensors = []
        for arg in args:
            result = ret_val.append_tensor(arg)
        if result is not None:
            # Then we did a dot product
            return result
        else:
            return ret_val

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Properties                                                                {{{1 |

    @property
    def external_indices(self):
        all_idxs = []
        for tens in self.tensors:
            all_idxs.extend(tens.indices)
        return tuple(a for a in all_idxs if all_idxs.count(a) == 1)

    @property
    def internal_indices(self):
        all_idxs = []
        for tens in self.tensors:
            all_idxs.extend(tens.indices)
        return tuple(
            sorted(
                list(set([a for a in all_idxs if all_idxs.count(a) == 2]))
            )
        )


    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Special Methods                                                           {{{1 |

    def __copy__(self):
        rv = EinsumContraction(*self.tensors)
        return rv

    def __neg__(self):
        return self.__mul__(-1.0)

    def __add__(self, other):
        if isinstance(other, (EinsumContraction, EinsumTensor)):
            return EinsumSum(self, other)
        else: # pragma: no cover
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (EinsumContraction, EinsumTensor)):
            return EinsumSum(self, -other)
        else: # pragma: no cover
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, EinsumTensor):
            result = self.append_tensor(other)
            if result is not None:
                return result
            else:
                return self
        elif isinstance(other, Real):
            rv = copy(self)
            rv.tensors[0].coeff *= other
            return rv
        else: # pragma: no cover
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Real):
            return self * other
        else: # pragma: no cover
            return NotImplemented

    def __str__(self):
        return " * ".join(str(t) for t in self.tensors)

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Methods                                                                   {{{1 |

    def append_tensor(self, tens):
        self.tensors.append(tens)
        if self.is_dot_product():
            return self.contract(None)
        else:
            return None

    def is_dot_product(self):
        return len(self.external_indices) == 0

    def contract(self, dest=None):
        # Gather a list of all the indices
        all_idxs = []
        for tens in self.tensors:
            all_idxs.extend(tens.indices)
        #========================================#
        # Check to make sure no index appears more than twice
        if sanity_checking_enabled:
            bad_idxs = [a for a in all_idxs if all_idxs.count(a) > 2]
            if len(bad_idxs) > 0:
                raise ValueError("too many appearances of index '{}' in contraction".format(bad_idxs[0]))
        #========================================#
        if dest is None:
            # for now, dest=None is only allowed in the case of a dot-product-like contraction
            dot_product = True
            if sanity_checking_enabled:
                bad_idxs = [a for a in all_idxs if all_idxs.count(a) != 2]
                if len(bad_idxs) > 0:
                    raise ValueError("index '{}' does not appear exactly twice in dot-product-like"
                                     " contraction with indices {}.".format(bad_idxs[0], all_idxs))
        else:
            dot_product = False
        #========================================#
        prefactor = reduce(mul, [t.coeff for t in self.tensors] , 1.0)
        interface = dest.interface
        #----------------------------------------#
        # TODO (semi-)automatic factorization of dot products
        if dot_product:
            # Create the index map
            indices = list(set(all_idxs))
            idxmap = {}
            for n, i in enumerate(indices):
                idxmap[i] = _index_set[n]
            #- - - - - - - - - - - - - - - - - - - - #
            # do the dot product
            dp_args = []
            for tens in self.tensors:
                dp_args.append(tens.sliced_tensor._impl)
                dp_args.append("".join(idxmap[i] for i in tens.indices))
            return interface.dot_product(prefactor, *dp_args)
        #----------------------------------------#
        elif len(self.tensors) == 2:
            # Create the index map
            indices = list(set(all_idxs))
            idxmap = {}
            for n, i in enumerate(indices):
                idxmap[i] = _index_set[n]
            dest.sliced_tensor._impl.contract_into(
                prefactor,
                self.tensors[0].sliced_tensor,
                tuple(idxmap[i] for i in self.tensors[0].indices),
                self.tensors[1].sliced_tensor,
                tuple(idxmap[i] for i in self.tensors[1].indices),
                1.0,
                tuple(idxmap[i] for i in dest.indices)
            )
            return dest.sliced_tensor
        #----------------------------------------#
        # Rough automatic factorization...
        else:
            #p = lambda x: None  # Initialize p so the code inspector will be happy
            #if self.print_factorization:
            #    file = self.print_factorization if self.print_factorization is not True else sys.stdout
            #    p = lambda x: print(x, file=file)
            #    p("Factorization of contraction {} <= {}".format(str(dest), str(self)))
            left = self.tensors[0]
            left.coeff = prefactor
            ltmp = None
            for i, right in enumerate(self.tensors[1:]):
                right.coeff = 1.0
                contr = EinsumContraction(left, right)
                if i < len(self.tensors) - 2:
                    out_idxs = contr.external_indices
                    #if self.print_factorization:
                    #    p('   {{intermediate {}}}["{}"] = {} * {}'.format(
                    #        i+1,  ",".join(out_idxs),  str(left),  str(right)))
                    out_shape = []
                    for idx in out_idxs:
                        if idx in left.indices:
                            out_shape.append(left.shape[left.indices.index(idx)])
                        else:
                            out_shape.append(right.shape[right.indices.index(idx)])
                    # The new left tensor will be the contraction intermediate
                    old_ltmp = ltmp
                    ltmp = interface.create_tensor(out_shape)
                    left = EinsumTensor(
                        indices=out_idxs,
                        tensor=Tensor(
                            indices=out_idxs,
                            index_range_set=dest._tensor.index_range_set,
                            _impl=ltmp,
                            interface=interface
                        ),
                        coeff=1.0,
                        index_range_set=dest._tensor.index_range_set,
                        known_indices=True
                    )
                    left._tensor.name = "{{intermediate {}}}".format(i+1)
                    contr.contract(dest=left)
                    # release the previous temporary
                    if old_ltmp is not None:
                        interface.release_tensor(old_ltmp)
                else:
                    #if self.print_factorization:
                    #    p('    {} = {} * {} '.format(
                    #        str(dest), str(left), str(right)))
                    contr.contract(dest=dest)
            # release the temporary
            interface.release_tensor(ltmp)
            # Return the sliced tensor, just like we do above
            rv = dest.sliced_tensor
            #========================================#
        return rv

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    pass
    # end class EinsumContraction

class EinsumSum(object):
    """
    Sum of EinsumProducts and EinsumTensors
    """

    #--------------------------------------------------------------------------------#

    #region | Attributes                                                                {{{1 |

    summands = None

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Initialization                                                            {{{1 |

    def __init__(self, *args):
        self.summands = args
        #----------------------------------------#
        new_summands = []
        # Flatten any inner sums
        for term in self.summands:
            if isinstance(term, EinsumSum):
                new_summands.extend(term.summands)
            else:
                new_summands.append(term)
        self.summands = new_summands

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Special Methods                                                           {{{1 |

    def __neg__(self):
        new_terms = []
        for term in self.summands:
            new_terms.append(-term)
        return EinsumSum(new_terms)

    def __add__(self, other):
        if isinstance(other, (EinsumTensor, EinsumContraction)):
            self.summands.append(other)
            return self
        elif isinstance(other, EinsumSum):
            self.summands.extend(other.summands)
            return self
        else: # pragma: no cover
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (EinsumTensor, EinsumContraction)):
            self.summands.append(-other)
            return self
        elif isinstance(other, EinsumSum):
            self.summands.extend([-o for o in other.summands])
            return self
        else: # pragma: no cover
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Real):
            for term in self.summands:
                term.coeff *= other
            return self
        else: # pragma: no cover
            return NotImplemented

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Methods                                                                   {{{1 |

    def sum_into(self, dest, accumulate=False):
        # TODO Check if dest is anywhere on the right hand side and copy if necessary
        #----------------------------------------#
        dst_slice = dest.sliced_tensor
        for term in self.summands:
            #----------------------------------------#
            # add contraction term
            if isinstance(term, EinsumContraction):
                if set(dest.indices) != set(term.external_indices):
                    raise EinsteinSummationAlignmentError("mismatched lhs/rhs"
                                                          " indices: ({}) != ({})".format(
                        ", ".join(dest.indices),
                        ", ".join(term.external_indices),
                    ))
                #- - - - - - - - - - - - - - - - - - - - #
                idx_range_set = dest.index_range_set
                interface = dest.interface
                # This is more complicated that it needs to be since we don't
                #   have broadcasting
                out_idxs = sorted(term.external_indices, key=dest.indices.index)
                out_shape = tuple(idx_range_set[i] for i in out_idxs)
                impl_tmp = interface.create_tensor(out_shape)
                contr_tmp = EinsumTensor(
                    indices=out_idxs,
                    tensor=Tensor(
                        indices=out_idxs,
                        index_range_set=dest._tensor.index_range_set,
                        _impl=impl_tmp,
                        interface=interface
                    ),
                    coeff=1.0,
                    index_range_set=dest._tensor.index_range_set,
                    known_indices=True
                )
                term.contract(contr_tmp)
                dst_slice._impl.add_into(1.0, contr_tmp._impl,
                    1.0 if accumulate or term is not self.summands[0] else 0.0
                )
                interface.release_tensor(impl_tmp)
            #----------------------------------------#
            # add individual tensor term
            else:
                if set(dest.indices) != set(term.indices):
                    raise EinsteinSummationAlignmentError("mismatched lhs/rhs"
                                                          " indices: ({}) != ({})".format(
                        ", ".join(dest.indices),
                        ", ".join(term.indices),
                    ))
                #- - - - - - - - - - - - - - - - - - - - #
                dst_slice._impl.add_into(1.0, term.sliced_tensor._impl,
                    1.0 if accumulate or term is not self.summands[0] else 0.0
                )
            #----------------------------------------#
        return dst_slice



    #endregion }}}1

    #--------------------------------------------------------------------------------#

    pass
    # end class EinsumSum


from tensor import Tensor
