#
# Copyright (C) 2016 University of Oxford
#
# This file is part of msprime.
#
# msprime is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# msprime is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with msprime.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Test cases for stats calculations in msprime.
"""
from __future__ import print_function
from __future__ import division

import unittest

import numpy as np

import msprime
import _msprime
import tests.tsutil as tsutil
import tests.test_wright_fisher as wf


def get_r2_matrix(ts):
    """
    Returns the matrix for the specified tree sequence. This is computed
    via a straightforward Python algorithm.
    """
    n = ts.get_sample_size()
    m = ts.get_num_mutations()
    A = np.zeros((m, m), dtype=float)
    for t1 in ts.trees():
        for sA in t1.sites():
            assert len(sA.mutations) == 1
            mA = sA.mutations[0]
            A[sA.id, sA.id] = 1
            fA = t1.get_num_samples(mA.node) / n
            samples = list(t1.samples(mA.node))
            for t2 in ts.trees(tracked_samples=samples):
                for sB in t2.sites():
                    assert len(sB.mutations) == 1
                    mB = sB.mutations[0]
                    if sB.position > sA.position:
                        fB = t2.get_num_samples(mB.node) / n
                        fAB = t2.get_num_tracked_samples(mB.node) / n
                        D = fAB - fA * fB
                        r2 = D * D / (fA * fB * (1 - fA) * (1 - fB))
                        A[sA.id, sB.id] = r2
                        A[sB.id, sA.id] = r2
    return A


class TestLdCalculator(unittest.TestCase):
    """
    Tests for the LdCalculator class.
    """

    num_test_sites = 50

    def verify_matrix(self, ts):
        m = ts.get_num_sites()
        ldc = msprime.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        self.assertEqual(A.shape, (m, m))
        B = get_r2_matrix(ts)
        self.assertTrue(np.allclose(A, B))

        # Now look at each row in turn, and verify it's the same
        # when we use get_r2 directly.
        for j in range(m):
            a = ldc.get_r2_array(j, direction=msprime.FORWARD)
            b = A[j, j + 1:]
            self.assertEqual(a.shape[0], m - j - 1)
            self.assertEqual(b.shape[0], m - j - 1)
            self.assertTrue(np.allclose(a, b))
            a = ldc.get_r2_array(j, direction=msprime.REVERSE)
            b = A[j, :j]
            self.assertEqual(a.shape[0], j)
            self.assertEqual(b.shape[0], j)
            self.assertTrue(np.allclose(a[::-1], b))

        # Now check every cell in the matrix in turn.
        for j in range(m):
            for k in range(m):
                self.assertAlmostEqual(ldc.get_r2(j, k), A[j, k])

    def verify_max_distance(self, ts):
        """
        Verifies that the max_distance parameter works as expected.
        """
        mutations = list(ts.mutations())
        ldc = msprime.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        j = len(mutations) // 2
        for k in range(j):
            x = mutations[j + k].position - mutations[j].position
            a = ldc.get_r2_array(j, max_distance=x)
            self.assertEqual(a.shape[0], k)
            self.assertTrue(np.allclose(A[j, j + 1: j + 1 + k], a))
            x = mutations[j].position - mutations[j - k].position
            a = ldc.get_r2_array(j, max_distance=x, direction=msprime.REVERSE)
            self.assertEqual(a.shape[0], k)
            self.assertTrue(np.allclose(A[j, j - k: j], a[::-1]))
        L = ts.get_sequence_length()
        m = len(mutations)
        a = ldc.get_r2_array(0, max_distance=L)
        self.assertEqual(a.shape[0], m - 1)
        self.assertTrue(np.allclose(A[0, 1:], a))
        a = ldc.get_r2_array(m - 1, max_distance=L, direction=msprime.REVERSE)
        self.assertEqual(a.shape[0], m - 1)
        self.assertTrue(np.allclose(A[m - 1, :-1], a[::-1]))

    def verify_max_mutations(self, ts):
        """
        Verifies that the max mutations parameter works as expected.
        """
        mutations = list(ts.mutations())
        ldc = msprime.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        j = len(mutations) // 2
        for k in range(j):
            a = ldc.get_r2_array(j, max_mutations=k)
            self.assertEqual(a.shape[0], k)
            self.assertTrue(np.allclose(A[j, j + 1: j + 1 + k], a))
            a = ldc.get_r2_array(j, max_mutations=k, direction=msprime.REVERSE)
            self.assertEqual(a.shape[0], k)
            self.assertTrue(np.allclose(A[j, j - k: j], a[::-1]))

    def test_single_tree_simulated_mutations(self):
        ts = msprime.simulate(20, mutation_rate=10, random_seed=15)
        ts = tsutil.subsample_sites(ts, self.num_test_sites)
        self.verify_matrix(ts)
        self.verify_max_distance(ts)

    def test_deprecated_aliases(self):
        ts = msprime.simulate(20, mutation_rate=10, random_seed=15)
        ts = tsutil.subsample_sites(ts, self.num_test_sites)
        ldc = msprime.LdCalculator(ts)
        A = ldc.get_r2_matrix()
        B = ldc.r2_matrix()
        self.assertTrue(np.array_equal(A, B))
        a = ldc.get_r2_array(0)
        b = ldc.r2_array(0)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(ldc.get_r2(0, 1), ldc.r2(0, 1))

    def test_single_tree_regular_mutations(self):
        ts = msprime.simulate(self.num_test_sites, length=self.num_test_sites)
        ts = tsutil.insert_branch_mutations(ts)
        # We don't support back mutations, so this should fail.
        self.assertRaises(_msprime.LibraryError, self.verify_matrix, ts)
        self.assertRaises(_msprime.LibraryError, self.verify_max_distance, ts)

    def test_tree_sequence_regular_mutations(self):
        ts = msprime.simulate(
            self.num_test_sites, recombination_rate=1,
            length=self.num_test_sites)
        self.assertGreater(ts.get_num_trees(), 10)
        t = ts.dump_tables()
        t.sites.reset()
        t.mutations.reset()
        for j in range(self.num_test_sites):
            site_id = len(t.sites)
            t.sites.add_row(position=j, ancestral_state="0")
            t.mutations.add_row(site=site_id, derived_state="1", node=j)
        ts = msprime.load_tables(**t.asdict())
        self.verify_matrix(ts)
        self.verify_max_distance(ts)

    def test_tree_sequence_simulated_mutations(self):
        ts = msprime.simulate(20, mutation_rate=10, recombination_rate=10)
        self.assertGreater(ts.get_num_trees(), 10)
        ts = tsutil.subsample_sites(ts, self.num_test_sites)
        self.verify_matrix(ts)
        self.verify_max_distance(ts)
        self.verify_max_mutations(ts)


def set_partitions(collection):
    """
    Returns an ierator over all partitions of the specified set.

    From https://stackoverflow.com/questions/19368375/set-partitions-in-python
    """
    if len(collection) == 1:
        yield [collection]
    else:
        first = collection[0]
        for smaller in set_partitions(collection[1:]):
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
            yield [[first]] + smaller


class TestMeanNumSamples(unittest.TestCase):
    """
    Tests the TreeSequence.mean_num_samples method.
    """
    def naive_mean_num_samples(self, ts, sample_sets):
        """
        Straightforward implementation of mean sample ancestry by iterating
        over the trees and nodes in each tree.
        """
        C = np.zeros((len(sample_sets), ts.num_nodes))
        T = np.zeros(ts.num_nodes)
        tree_iters = [ts.trees(tracked_samples=sample_set) for sample_set in sample_sets]
        for _ in range(ts.num_trees):
            trees = [next(tree_iter) for tree_iter in tree_iters]
            left, right = trees[0].interval
            length = right - left
            for node in trees[0].nodes():
                num_samples = trees[0].num_samples(node)
                if num_samples > 0:
                    for j, tree in enumerate(trees):
                        C[j, node] += length * tree.num_tracked_samples(node)
                    T[node] += length
        # Any nodes that are ancestral to zero samples have value zero.
        index = T != 0
        C[:, index] /= T[index]
        return C

    def verify(self, ts, sample_sets):
        C1 = self.naive_mean_num_samples(ts, sample_sets)
        C2 = tsutil.mean_num_samples(ts, sample_sets)
        self.assertEqual(C1.shape, C2.shape)
        self.assertTrue(np.allclose(C1, C2))
        return C1

    def test_two_populations_high_migration(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(8),
                msprime.PopulationConfiguration(8)],
            migration_matrix=[[0, 1], [1, 0]],
            recombination_rate=3,
            random_seed=5)
        self.assertGreater(ts.num_trees, 1)
        self.verify(ts, [ts.samples(0), ts.samples(1)])

    def test_single_tree(self):
        ts = msprime.simulate(6, random_seed=1)
        S = [range(3), range(3, 6)]
        C = self.verify(ts, S)
        for j, samples in enumerate(S):
            tree = next(ts.trees(tracked_samples=samples))
            for u in tree.nodes():
                self.assertEqual(tree.num_tracked_samples(u), C[j, u])

    def test_single_tree_partial_samples(self):
        ts = msprime.simulate(6, random_seed=1)
        S = [range(3), range(3, 4)]
        C = self.verify(ts, S)
        for j, samples in enumerate(S):
            tree = next(ts.trees(tracked_samples=samples))
            for u in tree.nodes():
                self.assertEqual(tree.num_tracked_samples(u), C[j, u])

    def test_single_tree_all_sample_sets(self):
        ts = msprime.simulate(6, random_seed=1)
        for S in set_partitions(list(range(ts.num_samples))):
            C = self.verify(ts, S)
            for j, samples in enumerate(S):
                tree = next(ts.trees(tracked_samples=samples))
                for u in tree.nodes():
                    self.assertEqual(tree.num_tracked_samples(u), C[j, u])

    def test_many_trees_all_sample_sets(self):
        ts = msprime.simulate(6, recombination_rate=2, random_seed=1)
        self.assertGreater(ts.num_trees, 2)
        for S in set_partitions(list(range(ts.num_samples))):
            self.verify(ts, S)

    def test_wright_fisher_unsimplified_all_sample_sets(self):
        tables = wf.wf_sim(
            4, 5, seed=1, deep_history=False, initial_generation_samples=False,
            num_loci=10)
        tables.sort()
        ts = tables.tree_sequence()
        for S in set_partitions(list(ts.samples())):
            self.verify(ts, S)

    def test_wright_fisher_unsimplified(self):
        tables = wf.wf_sim(
            20, 15, seed=1, deep_history=False, initial_generation_samples=False,
            num_loci=20)
        tables.sort()
        ts = tables.tree_sequence()
        samples = ts.samples()
        self.verify(ts, [samples[:10], samples[10:]])

    def test_wright_fisher_simplified(self):
        tables = wf.wf_sim(
            30, 10, seed=1, deep_history=False, initial_generation_samples=False,
            num_loci=5)
        tables.sort()
        ts = tables.tree_sequence()
        samples = ts.samples()
        self.verify(ts, [samples[:10], samples[10:]])
