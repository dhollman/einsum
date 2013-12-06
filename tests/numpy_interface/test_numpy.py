# -*- coding: utf-8 -*-
from __future__ import print_function, division
from itertools import product
from os.path import dirname, abspath, join as path_join
import sys
mydir = abspath(dirname(__file__))
sys.path.append(abspath(path_join(mydir, "..", "..")))
from tensors import Tensor
from tensors.numpy_interface import NumPyInterface
from tensors.numpy_interface.interface import _check_released
from tensors.indices import DeclareIndexRange, IndexRange, split_indices
from unittest import TestCase
import tensors
import unittest

tensors.type_checking_enabled = True
tensors.sanity_checking_enabled = True
Tensor.set_interface(NumPyInterface)

xproduct = lambda *args: product(*[xrange(*arg) if isinstance(arg, tuple) else xrange(arg) for arg in args])

# Note: in the future, we should have a not_implemented decorator
#   that requires a NotImplementedError to be raised
not_implemented = unittest.skip("not implemented")

class TestNumPyInterface(TestCase):

    interface = NumPyInterface

    def assertTensorEqual(self, first, second, msg=None):
        if hasattr(first, "_impl"):
            # allows handling of either Tensor or implementation types
            first, second = first._impl, second._impl
        #----------------------------------------#
        self.assertEqual(first.shape, second.shape,
            msg="{}shapes not equal: {} != {}".format(
                msg + "\n" if msg is not None else "",
                first.shape, second.shape
            )
        )
        #----------------------------------------#
        neq_idxs = []
        for idxs in xproduct(*first.shape):
            v1 = first.get_element(idxs)
            v2 = second.get_element(idxs)
            if v1 != v2:
                neq_idxs.append(idxs)
        self.assertEqual(len(neq_idxs), 0,
            msg="{}some values not equal:\n    {}".format(
                msg + "\n" if msg is not None else "",
                "\n    ".join(
                    "(" + ", ".join(str(i) for i in idxs) + "): "
                    + "{} != {}".format(first[idxs], second[idxs])
                        for idxs in neq_idxs
                )
            )
        )

    def setUp(self):
        self.my_created_tensors = []
        self.other_contexts = []
        #----------------------------------------#
        # create some sample ranges
        IndexRange.begin_indexing_context("sample_ranges")
        # Here we use the most straightforward version of the
        #   constructors.  Other versions are tested in specific
        #   tests for IndexRange.
        DeclareIndexRange("p,q,r,s,t,u", 0, 7, name="top").with_subranges(
            IndexRange("i,j,k,l", 0, 2, name="sub1"),
            IndexRange("a,b,c,d", 2, 7, name="sub2").with_subranges(
                IndexRange("a*,b*,c*,d*", 2, 6, name="subsub1").with_subranges(
                    IndexRange("a0,b0,c0,d0", 2, 2, name="zero sized subrange"),
                    IndexRange("a**,b**,c**,d**", 2, 6, name="trivial subrange"),
                ),
                IndexRange("a1,b1,c1,d1", 6, 7, name="unit subrange")
            )
        )

    def tearDown(self):
        #----------------------------------------#
        # release all of our tensors
        for t in self.my_created_tensors:
            self.interface.release_tensor(t)
        #----------------------------------------#
        # check to make sure no other tensors were leaked
        _check_released()
        #----------------------------------------#
        for context in self.other_contexts:
            IndexRange.end_indexing_context(context)
        IndexRange.end_indexing_context("sample_ranges")

    def make_trivial_range(self):
        IndexRange.begin_indexing_context("trivial")
        DeclareIndexRange("p,q,r,s,t,u", 0, 1, name="trivial")
        self.other_contexts.insert(0, "trivial")

    def make_small_range(self):
        IndexRange.begin_indexing_context("small")
        DeclareIndexRange("p,q,r,s,t,u", 0, 2, name="small")
        self.other_contexts.insert(0, "small")

    def make_range_tensor(self, indices, offset=0.0, factor=1.0):
        shape = tuple(
            IndexRange.global_index_range_set[idx].size
                for idx in split_indices(indices)
        )
        rv_impl = self.interface.create_tensor(shape)
        val = offset
        for idxs in xproduct(*shape):
            rv_impl.set_element(idxs, val)
            val += factor
        #----------------------------------------#
        rv = Tensor(indices, _impl=rv_impl, interface=type(self).interface)
        self.my_created_tensors.append(rv_impl)
        return rv

    def indices_iterator(self, indices, offset_starts=False, parent_indices=None):
        indices = split_indices(indices)
        rngs = [IndexRange.global_index_range_set[idx] for idx in indices]
        if any(idx == indices[0] for idx in indices[1:]):
            raise NotImplemented("repeated indices not implemented")
        if offset_starts and parent_indices is None: # offset in root range
            xrngs = [xrange(
                r.begin_index,
                r.end_index,
                r.slice.step if r.slice.step is not None else 1
            ) for r in rngs]
        elif parent_indices is None: # don't offset
            # default behavior
            xrngs = [xrange(r.size) for r in rngs]
        else: # do_offset is True, parent indices are given
            prngs = [IndexRange.global_index_range_set[idx] for idx in split_indices(parent_indices)]
            xrngs = [xrange(
                r.slice_in(pr).start,
                r.slice_in(pr).stop,
                r.slice_in(pr).step or 1
            ) for r, pr in zip(rngs, prngs)]
        return product(*xrngs)

    # it's a meta-test...
    def test_test_indices_iterator(self):
        self.assertEqual(
            list(self.indices_iterator("p,q,r,s")),
            list(xproduct(7, 7, 7, 7))
        )
        self.assertEqual(
            list(self.indices_iterator("i,j,k,l")),
            list(xproduct(2, 2, 2, 2))
        )
        self.assertEqual(
            list(self.indices_iterator("i,p,k,q")),
            list(xproduct(2, 7, 2, 7))
        )
        self.assertEqual(
            list(self.indices_iterator("i,p,k,q")),
            list(xproduct(2, 7, 2, 7))
        )
        self.assertEqual(
            list(self.indices_iterator("i,p,k,a0")),
            []
        )
        self.assertEqual(
            list(self.indices_iterator("i,p,k,a0", offset_starts=True)),
            []
        )
        self.assertEqual(
            list(self.indices_iterator("i,p,a,b1", offset_starts=True)),
            list(xproduct(2, 7, (2, 7), (6, 7)))
        )
        self.assertEqual(
            list(self.indices_iterator("i,p,a,b1", offset_starts=True, parent_indices="p,q,r,a")),
            list(xproduct(2, 7, (2, 7), (4, 5)))
        )

    def make_tensor(self, *args):
        """ thin wrapper that tracks creation for cleanup """
        rv = Tensor(*args)
        # TODO add some noise to make sure tests fail when e.g. adding when we should be overwriting
        self.my_created_tensors.append(rv._impl)
        return rv

    #--------------------------------------------------------------------------------#

    def test_sort_1(self):
        t1 = self.make_range_tensor("p,q,r,s")
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,s,q,r"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1 = expect._impl, t1._impl
        for p,q,r,s in self.indices_iterator("p,q,r,s"):
            E[p,q,r,s] = T1[p,s,q,r]
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_sort_self(self):
        t1 = self.make_range_tensor("p,q,r,s")
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,r,s"]
        result["p,q,r,s"] = result["p,s,q,r"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1 = expect._impl, t1._impl
        for p,q,r,s in self.indices_iterator("p,q,r,s"):
            E[p,q,r,s] = T1[p,s,q,r]
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    #--------------------------------------------------------------------------------#

    @not_implemented
    def test_add_constant(self):
        t1 = self.make_range_tensor("p,q,r,s")
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,r,s"] + 3.14
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        for idxs in xproduct(7, 7, 7, 7):
            expect._impl.set_element(idxs,
                t1._impl.get_element(idxs) + 3.14
            )
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_add_1(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q,r,s", offset=3.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,r,s"] + t2["p,q,r,s"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        for idxs in self.indices_iterator("p,q,r,s"):
            expect._impl.set_element(idxs,
                t1._impl.get_element(idxs)
                + t2._impl.get_element(idxs)
            )
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_add_2(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("a,p,b1,c**", offset=3.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["a,p,b1,c**"] = t1["a,p,b1,c**"] + t2["a,p,b1,c**"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        for idxs_t1, idxs_t2 in zip(
                self.indices_iterator("a,p,b1,c**", offset_starts=True),
                self.indices_iterator("a,p,b1,c**", offset_starts=False)):
            expect._impl.set_element(idxs_t1,
                t1._impl.get_element(idxs_t1)
                + t2._impl.get_element(idxs_t2)
            )
            #----------------------------------------#
        self.assertTensorEqual(result["a,p,b1,c**"]._, expect["a,p,b1,c**"]._)

    def test_add_3(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q,r,s", offset=3.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,r,s"] + t2["q,r,s,p"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        for i in self.indices_iterator("p,q,r,s"):
            expect._impl.set_element(i,
                t1._impl.get_element(i)
                + t2._impl.get_element((i[1],i[2],i[3],i[0]))
            )
            #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_add_4(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q,r,s", offset=3.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = 5.0 * t1["p,q,r,s"] + 3.0 * t2["q,r,s,p"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1, T2 = expect._impl, t1._impl, t2._impl
        for p,q,r,s in self.indices_iterator("p,q,r,s"):
            E[p,q,r,s] = 5.0 * T1[p,q,r,s] + 3.0 * T2[q,r,s,p]
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_add_multiple(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("a,b,p,s", offset=3.0)
        t3 = self.make_range_tensor("a1,b1,c**,i", offset=5.0)
        t4 = self.make_range_tensor("p,q,r,s", offset=8.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,r,s"] + t4["p,q,r,s"]
        result["a,b,p,s"] = result["a,b,p,s"] + t2["a,b,p,s"]
        result["a1,b1,c**,i"] = result["a1,b1,c**,i"] + t3["a1,b1,c**,i"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        for idxs in self.indices_iterator("p,q,r,s"):
            expect._impl.set_element(idxs,
                t1._impl.get_element(idxs)
                + t4._impl.get_element(idxs)
            )
        for idxs_1, idxs_2 in zip(
                self.indices_iterator("a,b,p,s", offset_starts=True),
                self.indices_iterator("a,b,p,s", offset_starts=False)):
            expect._impl.set_element(idxs_1,
                expect._impl.get_element(idxs_1)
                + t2._impl.get_element(idxs_2)
            )
        for idxs_1, idxs_2 in zip(
                self.indices_iterator("a1,b1,c**,i", offset_starts=True),
                self.indices_iterator("a1,b1,c**,i", offset_starts=False)):
            expect._impl.set_element(idxs_1,
                expect._impl.get_element(idxs_1)
                + t3._impl.get_element(idxs_2)
            )
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_add_self_small(self):
        self.make_small_range()
        t1 = self.make_range_tensor("p,q,r", offset=3.0)
        result = self.make_tensor("p,q,r")
        expect = self.make_tensor("p,q,r")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r"] = t1["p,q,r"]
        result["p,q,r"] = result["p,r,q"] + t1["p,q,r"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1 = expect._impl, t1._impl
        for p,q,r in self.indices_iterator("p,q,r"):
            E[p,q,r] = T1[p,r,q] + T1[p,q,r]
        #----------------------------------------#
        print(T1._array)
        self.assertTensorEqual(result, expect)

    def test_add_self(self):
        t1 = self.make_range_tensor("p,q,r,s")
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        etmp = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["s,q,p,r"] + t1["p,r,q,s"]
        result["p,r,q,s"] = result["p,s,q,r"] + t1["p,r,s,q"]
        result["p,r,q,s"] = result["p,s,q,r"] + result["s,r,q,p"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1, Ee = expect._impl, t1._impl, etmp._impl
        for p,q,r,s in self.indices_iterator("p,q,r,s"):
            E[p,q,r,s] = T1[s,q,p,r] + T1[p,r,q,s]
        for p,q,r,s in self.indices_iterator("p,q,r,s"): Ee[p,q,r,s] = E[p,q,r,s]
        for p,q,r,s in self.indices_iterator("p,q,r,s"):
            E[p,r,q,s] = Ee[p,s,q,r] + T1[p,r,s,q]
        for p,q,r,s in self.indices_iterator("p,q,r,s"): Ee[p,q,r,s] = E[p,q,r,s]
        for p,q,r,s in self.indices_iterator("p,q,r,s"):
            E[p,r,q,s] = Ee[p,s,q,r] + Ee[s,r,q,p]
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    #--------------------------------------------------------------------------------#

    def test_contract_1(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q,r,s", offset=3.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,t,u"] * t2["t,u,r,s"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1, T2 = expect._impl, t1._impl, t2._impl
        for i in self.indices_iterator("p,q,r,s"): E[i] = 0.0
        for p,q,r,s,t,u in self.indices_iterator("p,q,r,s,t,u"):
            E[p,q,r,s] = E[p,q,r,s] + T1[p,q,t,u] * T2[t,u,r,s]
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_contract_2(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q,r,s", offset=2.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,a,i"] * t2["i,a,s,r"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1, T2 = expect._impl, t1._impl, t2._impl
        for i in self.indices_iterator("p,q,r,s"): E[i] = 0.0
        for p,q,r,s,i,a in self.indices_iterator("p,q,r,s,i,a", offset_starts=True):
            E[p,q,r,s] = E[p,q,r,s] + T1[p,q,a,i] * T2[i,a,s,r]
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_contract_3(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q", offset=2.0)
        t3 = self.make_range_tensor("p,q", offset=5.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,a,i"] * t2["i,r"] * t3["a,s"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1, T2, T3 = expect._impl, t1._impl, t2._impl, t3._impl
        for i in self.indices_iterator("p,q,r,s"): E[i] = 0.0
        for p,q,r,s,i,a in self.indices_iterator("p,q,r,s,i,a", offset_starts=True):
            E[p,q,r,s] = E[p,q,r,s] + T1[p,q,a,i] * T2[i,r] * T3[a,s]
            #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_contract_4(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q", offset=2.0)
        t3 = self.make_range_tensor("p,q", offset=5.0)
        t4 = self.make_range_tensor("p,q,r", offset=5.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,a,i"] * t2["i,r"] * t2["a,b1"] * t3["b1,k"] * t4["k,a1,c1"] * t4["a1,c1,s"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1, T2, T3, T4 = expect._impl, t1._impl, t2._impl, t3._impl, t4._impl
        for i in self.indices_iterator("p,q,r,s"): E[i] = 0.0
        for p,q,r,s,i,a,a1,b1,c1,k in self.indices_iterator("p,q,r,s,i,a,a1,b1,c1,k", offset_starts=True):
            E[p,q,r,s] = E[p,q,r,s] + T1[p,q,a,i] * T2[i,r] * T2[a,b1] * T3[b1,k] * T4[k,a1,c1] * T4[a1,c1,s]
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    #--------------------------------------------------------------------------------#

    # TODO Test dot products and traces

    def test_contract_sum(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q,r,s", offset=2.0)
        t3 = self.make_range_tensor("p,q,r,s", offset=3.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = t1["p,q,a,i"] * t2["i,a,s,r"] - t3["p,a1,q,c**"] * t1["a1,s,r,c**"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1, T2, T3 = expect._impl, t1._impl, t2._impl, t3._impl
        for i in self.indices_iterator("p,q,r,s"): E[i] = 0.0
        for p,q,r,s,i,a in self.indices_iterator("p,q,r,s,i,a", offset_starts=True):
            E[p,q,r,s] = E[p,q,r,s] + T1[p,q,a,i] * T2[i,a,s,r]
        for p,q,r,s,a1,css in self.indices_iterator("p,q,r,s,a1,c**", offset_starts=True):
            E[p,q,r,s] = E[p,q,r,s] - T3[p,a1,q,css] * T1[a1,s,r,css]
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    def test_contract_sum_2(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q,r,s", offset=2.0)
        t3 = self.make_range_tensor("p,q,r,s", offset=3.0)
        result = self.make_tensor("p,q,r,s")
        expect = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = 13.0 * t1["p,q,a,i"] * t2["i,a,s,r"] - 5.0 * t3["p,a1,q,c**"] * t1["a1,s,r,c**"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, T1, T2, T3 = expect._impl, t1._impl, t2._impl, t3._impl
        for i in self.indices_iterator("p,q,r,s"): E[i] = 0.0
        for p,q,r,s,i,a in self.indices_iterator("p,q,r,s,i,a", offset_starts=True):
            E[p,q,r,s] = E[p,q,r,s] + 13.0 * T1[p,q,a,i] * T2[i,a,s,r]
        for p,q,r,s,a1,css in self.indices_iterator("p,q,r,s,a1,c**", offset_starts=True):
            E[p,q,r,s] = E[p,q,r,s] - 5.0 * T3[p,a1,q,css] * T1[a1,s,r,css]
            #----------------------------------------#
        self.assertTensorEqual(result, expect)

    # TODO for interfaces that actually implement shares_data_with(), more extensive tests involving (overlapping and non-overlapping) slices on left and right are needed
    def test_contract_sum_self(self):
        result = self.make_range_tensor("p,q,r,s")
        expect = self.make_range_tensor("p,q,r,s")
        expect_tmp = self.make_tensor("p,q,r,s")
        #----------------------------------------#
        # Einstein summation version
        result["p,q,r,s"] = \
            13.0 * result["p,q,a,i"] * result["i,a,s,r"] \
            - 5.0 * result["p,a1,c**,q"] * result["a1,s,r,c**"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        E, Ee = expect._impl, expect_tmp._impl
        for i in self.indices_iterator("p,q,r,s"): Ee[i], E[i] = E[i], 0.0
        for p,q,r,s,i,a in self.indices_iterator("p,q,r,s,i,a", offset_starts=True):
            E[p,q,r,s] = E[p,q,r,s] + 13.0 * Ee[p,q,a,i] * Ee[i,a,s,r]
        for p,q,r,s,a1,css in self.indices_iterator("p,q,r,s,a1,c**", offset_starts=True):
            E[p,q,r,s] = E[p,q,r,s] - 5.0 * Ee[p,a1,css,q] * Ee[a1,s,r,css]
        #----------------------------------------#
        self.assertTensorEqual(result, expect)

    #--------------------------------------------------------------------------------#

    def test_dot_1(self):
        t1 = self.make_range_tensor("p,q,r,s")
        t2 = self.make_range_tensor("p,q,r,s", offset=3.0)
        expect = 0
        #----------------------------------------#
        # Einstein summation version
        result = t1["p,q,r,s"] * t2["s,q,p,r"]
        #----------------------------------------#
        # The same thing, in terms of get_element()
        #   and set_element() calls
        T1, T2 = t1._impl, t2._impl
        for p,q,r,s in self.indices_iterator("p,q,r,s"):
            expect += T1[p,q,r,s] * T2[s,q,p,r]
        #----------------------------------------#
        self.assertEqual(result, expect)

