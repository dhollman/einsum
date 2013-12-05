# -*- coding: utf-8 -*-
"""
Classes for creating and managing indices of tensors.
"""
from __future__ import division, print_function
import re
from tensors import EinsteinSummationIndexingError, sanity_checking_enabled, type_checking_enabled

def is_subrange(child_range, parent_range):
    """
    Returns True if `child_range` is `parent_range` or `child_range` is a child of `parent_range`
    (analogous to `isinstance` from the python standard library)
    """
    if child_range is parent_range:
        return True
    parents = []
    spot = child_range
    while spot.parent is not None:
        parents.append(spot.parent)
        spot = spot.parent
    return parent_range in parents
issubrange = is_subrange

def DeclareIndexRange(indices, begin_index_or_size, end_index=None, name=None, **kwargs):
    """ Alias for `IndexRange` constructor that more accurately describes what is actually going on.
    """
    return IndexRange(indices, begin_index_or_size, end_index, name, **kwargs)

class IndexRangeSet(object):
    """
    A set of index ranges that work together to describe how indices are interpreted
    in a given context.
    """

    #--------------------------------------------------------------------------------#

    #region | Attributes                                                                {{{1 |

    known_ranges = None
    """
    Mapping from index string to `IndexRange` object for all indices and ranges in
    the current set.
    """

    name = None
    """
    An optional identifier, used for consistency checking in begin and end context calls
    """

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Initialization                                                            {{{1 |

    def __init__(self, name=None):
        self.known_ranges = dict()
        self.name = name

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Special Methods                                                           {{{1 |

    def __getitem__(self, item):
        return self.known_ranges[item]

    def __setitem__(self, key, item):
        self.known_ranges[key] = item

    def __contains__(self, item):
        if item in self.known_ranges:
            return True
        elif any(v.name == item for v in self.known_ranges.values()):
            return True
        else:
            return False

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    pass
    # end class IndexRangeSet

# Useful alias when used as a transparent context
IndexingContext = IndexRangeSet


class IndexRange(object):
    """ Allows for the definition of special indices that cover specific ranges and subranges of Tensor axes
    in einstein summations.

    :Examples:

    >>> IndexRange.clear_known_ranges()  # Don't do this in your program; this is just for the doctest
    >>> p = IndexRange('p,q,r,s', 5).with_subranges(
    ...   IndexRange('i,j,k,l', 0, 2),
    ...   IndexRange('a,b,c,d', 2, 5)
    ... )
    >>> p
    <IndexRange object covering slice 0:5 represented by indices ['p', 'q', 'r', 's']>
    >>> p.subranges[0]
    <IndexRange object covering slice 0:2 represented by indices ['i', 'j', 'k', 'l']>
    >>> p.subranges[1]
    <IndexRange object covering slice 2:5 represented by indices ['a', 'b', 'c', 'd']>
    """

    #--------------------------------------------------------------------------------#

    #region | Static Methods                                                            {{{1 |

    @staticmethod
    def split_indices(in_indices):
        """
        Split a string or tuple of indices into component parts.
        """
        rv_list = []
        # Note: basestring is not Python3 compatible!
        if isinstance(in_indices, basestring):
            idxs = tuple(re.split(r'\s*,\s*', in_indices))
        else:
            idxs = tuple(in_indices)
        #----------------------------------------#
        if len(idxs) > 1:
            for idx in idxs:
                rv_list.extend(
                    list(IndexRange.split_indices(idx))
                )
        elif len(idxs) == 1:
            rv_list = idxs
        else: # len(idxs) == 0
            raise EinsteinSummationIndexingError("empty index")
        #----------------------------------------#
        return tuple(rv_list)

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Class Attributes                                                          {{{1 |

    global_index_range_set = IndexRangeSet()
    """
    Represents the current indexing context
    """

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Private Class Attributes                                                  {{{1 |

    _global_index_context_stack = []
    """
    Suspended indexing contexts to be reactivated when current context ends.
    """

    _named_range_sets = dict()
    _range_set_names = dict()

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Class Methods                                                             {{{1 |

    @classmethod
    def clear_known_ranges(cls):
        """ Clears the known ranges.  Use with care and only if you're sure you know what you're doing!
        """
        cls.global_index_range_set.known_ranges = {}
    reset_ranges = clear_known_ranges

    @classmethod
    def set_global_index_range_set(cls, new_global_set_or_name, name_in=None):
        #----------------------------------------#
        if isinstance(new_global_set_or_name, IndexRangeSet):
            new_global_set = new_global_set_or_name
            if name_in is not None:
                new_global_set.name = name_in
            else:
                name_in = new_global_set.name
        #----------------------------------------#
        else:
            if name_in is not None:
                raise TypeError("first parameter of two-parameter form must be an IndexRangeSet")
            name_in = str(new_global_set_or_name)
            if name_in in cls._named_range_sets:
                raise EinsteinSummationIndexingError(
                    "IndexRangeSet named '{}' already exists".format(name_in)
                )
            new_global_set = IndexRangeSet(name=name_in)
        if name_in is not None:
            cls._named_range_sets[name_in] = new_global_set
        #----------------------------------------#
        cls._global_index_context_stack.append(cls.global_index_range_set)
        cls.global_index_range_set = new_global_set
        #----------------------------------------#
    # Aliases illustrating the better way to think about IndexRangeSets as a transparent context
    set_indexing_context = set_global_index_range_set
    begin_indexing_context = set_global_index_range_set

    @classmethod
    def unset_global_index_range_set(cls, context_to_end_or_name=None):
        # check to make sure there's a context to take it's place
        if len(cls._global_index_context_stack) == 0:
            raise ValueError("No indexing context to end.")
        #----------------------------------------#
        # Allow the user to give a context as a safety check
        context_to_end = context_to_end_or_name
        if context_to_end_or_name is not None and not isinstance(context_to_end_or_name, IndexRangeSet):
            name = str(context_to_end_or_name)
            if name not in cls._named_range_sets:
                raise EinsteinSummationIndexingError(
                    "unknown range named '{}' can't be ended".format(name)
                )
            context_to_end = cls._named_range_sets[name]
        if context_to_end is not None and cls.global_index_range_set is not context_to_end:
            name_end = context_to_end.name
            if name_end is None: name_end = "(unnamed context)"
            name_active = cls.global_index_range_set.name
            if name_active is None: name_active = "(unnamed context)"
            #- - - - - - - - - - - - - - - - - - - - #
            raise EinsteinSummationIndexingError(
                "requested end of indexing context '{}', but currently active"
                " context is '{}'{}".format(
                    name_end, name_active,
                    " (which does not refer to the same context)" if name_end == name_active else ""
                )
            )
        if context_to_end is not None and context_to_end.name is not None:
            cls._named_range_sets.pop(context_to_end.name)
        #----------------------------------------#
        cls.global_index_range_set = cls._global_index_context_stack.pop()
    # Alias illustrating the better way to think about IndexRangeSets as a transparent context
    end_indexing_context = unset_global_index_range_set

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Attributes                                                                {{{1 |

    indices = None
    """
    Tuple of the indices that can be used to describe the range `self`
    """

    subranges = None
    """
    List of immediate child ranges.
    """

    index_range_set = None
    """
    The IndexRangeSet that `self` belongs to.
    """

    name = None
    """
    A description of the range covered by the indices.  Useful for debugging.
    """

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Private Attributes                                                        {{{1 |

    _parent = None
    _begin_is_ellipsis = None
    _end_is_ellipsis = None
    _slice = None

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Initialization                                                            {{{1 |

    def __init__(self, indices,
            begin_index_or_size_or_slices,
            end_index=None,
            name=None,
            parent=None,
            index_range_set=None):
        #----------------------------------------#
        self.indices = split_indices(indices)
        self.subranges = []
        self.name = name
        self._begin_is_ellipsis = False
        self._end_is_ellipsis = False
        #----------------------------------------#
        if index_range_set is not None:
            self.index_range_set = index_range_set
        else:
            self.index_range_set = IndexRange.global_index_range_set
        #----------------------------------------#
        if isinstance(begin_index_or_size_or_slices, int):
            if end_index is not None:
                # then we're doing a range of indices
                if end_index is Ellipsis:
                    self._end_is_ellipsis = True
                    # The slice should run to the end of the parent, or the end of the
                    #   tensor's axis if it has no parent.
                    self.slice = slice(begin_index_or_size_or_slices, None)
                else:
                    # We've been given a begin and an end.  This is straightforward
                    self.slice = slice(begin_index_or_size_or_slices, end_index)
            else:
                # The number given to us is functioning as a size...
                self.slice = slice(0, begin_index_or_size_or_slices)
        #- - - - - - - - - - - - - - - - - - - - #
        elif isinstance(begin_index_or_size_or_slices, slice):
            # We've been given a slice.  Also straightforward
            self.slice = begin_index_or_size_or_slices
        #- - - - - - - - - - - - - - - - - - - - #
        elif begin_index_or_size_or_slices is Ellipsis:
            self._begin_is_ellipsis = True
            if isinstance(end_index, int):
                self.slice = slice(None, end_index)
            else:
                raise TypeError("Unsupported type for third argument to"
                                " IndexRange constructor of type {0}".format(
                    type(end_index).__name__)
                )
        #- - - - - - - - - - - - - - - - - - - - #
        else:
            raise TypeError("Unsupported type for second argument to"
                            " IndexRange constructor: {0}".format(
                type(begin_index_or_size_or_slices).__name__)
            )
        #----------------------------------------#
        # Now initialize any Ellipses
        self.parent = parent
        #----------------------------------------#
        for idx in self.indices:
            if idx in self.index_range_set.known_ranges:
                raise EinsteinSummationIndexingError(
                    u"Index {0} is already part of index range {1}.".format(
                        idx, self.index_range_set.known_ranges[idx]
                    )
                )
            self.index_range_set[idx] = self


    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Properties                                                                {{{1 |

    @property
    def begin_index(self):
        if self._slice.start is None:
            return 0
        elif self._slice.start < 0:
            raise EinsteinSummationIndexingError(
                "no definite out-of-context begin_index for negative slice component"
            )
        else:
            return self._slice.start

    @begin_index.setter
    def begin_index(self, idx):
        if idx is not None and idx > self.end_index and self.end_index >= 0:
            raise EinsteinSummationIndexingError(
                "new begin_index {0} is after end_index {1}".format(idx, self.end_index)
            )
        #----------------------------------------#
        self.slice = slice(idx, self._slice.stop, self._slice.step)

    @property
    def end_index(self):
        if self._slice.start is None or self._slice.start < 0:
            raise EinsteinSummationIndexingError(
                "no definite out-of-context end_index for negative slice component"
            )
        else:
            return self._slice.stop

    @end_index.setter
    def end_index(self, idx):
        if idx is not None and idx < self.begin_index and idx >= 0:
            raise EinsteinSummationIndexingError(
                "new end_index {0} is before begin_index {1}".format(idx, self.begin_index))
        self.slice = slice(self._slice.start, idx, self._slice.step)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, new_parent):
        if new_parent is None and self._parent is not None:
            raise EinsteinSummationIndexingError("can't unparent IndexRange objects")
        #----------------------------------------#
        if self._begin_is_ellipsis:
            if new_parent is None:
                self._slice.start = 0
            else:
                self.begin_index = new_parent.begin_index
        if self._end_is_ellipsis:
            if new_parent is not None:
                self.end_index = new_parent.end_index
        #----------------------------------------#
        self._parent = new_parent

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, new_slice):
        if type_checking_enabled:
            if not isinstance(new_slice, slice):
                raise TypeError("IndexRange.slice must an instance of 'slice'")
        if sanity_checking_enabled:
            if not new_slice.step is None and not new_slice.step == 1:
                raise NotImplementedError("IndexRange objects for slices with non-unitary steps are not yet implemented.")
        #----------------------------------------#
        self._slice = new_slice

    @property
    def size(self):
        try:
            return self.end_index - self.begin_index
        except EinsteinSummationIndexingError:
            raise EinsteinSummationIndexingError(
                "IndexRange with slice = {} does not have a known size out of context".format(
                    self.slice
                )
            )

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Special Methods                                                           {{{1 |

    def __len__(self):
        return self.size

    def __str__(self):
        if self.name is None:
            return "<IndexRange object covering slice {0.begin_index}:{0.end_index}" \
                   " represented by indices ['{1}']>".format(self, "', '".join(self.indices))
        else:
            return "<IndexRange named '{0.name}'" \
                   " represented by indices ['{1}']>".format(self, "', '".join(self.indices))
    __repr__ = __str__

    #endregion }}}1

    #--------------------------------------------------------------------------------#

    #region | Methods                                                                   {{{1 |

    def add_subrange(self, subrange):
        """ Add a subrange to the list of subranges for `self`.
        Note that this returns `subrange` (with `subrange.parent` modified) to allow for a 'pass-through' like usage
        (see examples below)

        :Examples:

        >>> IndexRange.reset_ranges()
        >>> p = IndexRange('p,q,r,s', 4, name="orbital space")
        >>> # Convenient "pass-through" usage for saving subranges:
        >>> i = p.add_subrange(IndexRange('i,j,k,l', 0, 2, name="occupied space"))
        >>> a = p.add_subrange(IndexRange('a,b,c,d',2,4))
        >>> a
        <IndexRange object covering slice 2:4 represented by indices ['a', 'b', 'c', 'd']>
        >>> p.subranges[0]
        <IndexRange named 'occupied space' represented by indices ['i', 'j', 'k', 'l']>
        >>> p.subranges[1]
        <IndexRange object covering slice 2:4 represented by indices ['a', 'b', 'c', 'd']>
        >>> a.parent
        <IndexRange named 'orbital space' represented by indices ['p', 'q', 'r', 's']>

        """
        if isinstance(subrange, IndexRange):
            if not subrange._begin_is_ellipsis and subrange.begin_index < self.begin_index:
                raise EinsteinSummationIndexingError(
                    "Subrange falls outside of parent range:  Subrange start ("
                         + str(subrange.begin_index) + ") is before parent range start ("
                         + str(self.begin_index) + ")"
                )
            elif not subrange._end_is_ellipsis and subrange.end_index > self.end_index:
                raise EinsteinSummationIndexingError(
                    "Subrange falls outside of parent range:  End of subrange"
                    " {} (end_index = {}) is after end of parent {} (end_index = {})".format(
                        str(subrange), subrange.end_index,
                        str(self), self.end_index
                    )
                )
            else:
                self.subranges.append(subrange)
                subrange.parent = self
                # Pass though for easy assigning...
                return subrange
        else:
            raise TypeError("subrange must be an instance of IndexRange")

    def with_subranges(self, *subranges):
        """ Basically the same thing as calling `add_subrange()` multiple times, except returns `self` instead of
        the subrange, allowing for a different syntax (see below)

        :Examples:

        >>> IndexRange.reset_ranges()
        >>> orb = DeclareIndexRange('p,q,r,s', 10, name="Orbital space").with_subranges(
        ...           DeclareIndexRange('i,j,k,l', 0, 3, name="Occupied space").with_subranges(
        ...               DeclareIndexRange("i*,j*,k*,l*", 0, 1, name="Core"),
        ...               DeclareIndexRange("i',j',k',l'", 1, 3)
        ...           ),
        ...           DeclareIndexRange('a,b,c,d', 3, 10, name="Virtual space")
        ...       )
        >>> orb
        <IndexRange named 'Orbital space' represented by indices ['p', 'q', 'r', 's']>
        >>> len(orb.subranges)
        2
        >>> len(orb.subranges[0].subranges)
        2

        """
        if sanity_checking_enabled:
            for subrange in subranges:
                if subrange.index_range_set is not self.index_range_set:
                    raise EinsteinSummationIndexingError(
                        "subrange is not part of the same IndexRangeSet as parent")
        #----------------------------------------#
        for subrange in subranges:
            self.add_subrange(subrange)
        return self

    def slice_in(self, parent_range):
        """ Gets the slice of `parent_range` represented by self
        (`parent_range` need not be a *direct* parent of self, but it should be a parent.  See `is_subrange()`)

        :Examples:

        >>> IndexRange.reset_ranges()
        >>> p = IndexRange('p,q,r,s',4)
        >>> i = p.add_subrange(IndexRange('i,j,k,l',0,2))
        >>> a = p.add_subrange(IndexRange('a,b,c,d',2,4))
        >>> a.slice_in(p)
        slice(2, 4, None)

        """
        if type_checking_enabled:
            if not isinstance(parent_range, IndexRange):
                raise TypeError("invalid argument type '{}'".format(type(parent_range).__name__))
        if sanity_checking_enabled:
            if not is_subrange(self, parent_range):
                raise EinsteinSummationIndexingError("range '{}' is not a subrange of '{}'".format(
                    self, parent_range
                ))
        #----------------------------------------#
        if parent_range is self:
            return slice(0, self.size)
        else:
            start = self.begin_index - parent_range.begin_index
            return slice(start, start + self.size)


    #--------------------------------------------------------------------------------#

    pass
    # end class IndexRange


# Useful alias
split_indices = IndexRange.split_indices
