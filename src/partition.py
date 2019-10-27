import operator
from functools import reduce
from itertools import starmap, zip_longest

import numpy as np
from cached_property import cached_property


class Partition:
    def __init__(self, *args):
        args = args or (0,)

        if isinstance(args[0], Partition):
            self.parts = args[0].parts  # already a tuple by construction

        elif isinstance(args[0], (list, tuple)):
            self.parts = tuple(sorted([x for x in args[0] if x], reverse=True))  # cast as tuple

        else:
            self.parts = tuple(sorted((int(x) for x in args if x), reverse=True))  # cast as tuple

        self.iter_index = -1
        self.end = len(self.parts)

    def __bool__(self):
        return bool(self.parts)

    def __len__(self):
        return len(self.parts)

    def __eq__(self, other):
        if isinstance(other, Partition):
            return self.parts == other.parts

        else:
            return self.parts == other

    def __lt__(self, other):
        return self.parts < other.parts

    def __le__(self, other):
        return self.parts <= other.parts

    def __gt__(self, other):
        return self.parts > other.parts

    def __ge__(self, other):
        return self.parts >= other.parts

    def __repr__(self):
        return str(self.parts)

    def __str__(self):
        """ U25A2 """
        return "\n".join("▢" * part for part in self.parts)

    def __add__(self, other):
        if isinstance(other, (Partition)):
            both = zip_longest(self.parts, other.parts, fillvalue=0)

        elif isinstance(other, (list, tuple)):
            both = zip_longest(self.parts, other, fillvalue=0)

        elif isinstance(other, int):
            return Partition((part + other for part in self.parts))

        return Partition(tuple(starmap(operator.add, both)))

    def __sub__(self, other):
        if isinstance(other, (Partition)):
            both = zip_longest(self.parts, other.parts, fillvalue=0)

        elif isinstance(other, (list, tuple)):
            both = zip_longest(self.parts, other, fillvalue=0)

        elif isinstance(other, int):
            return Partition((part + other for part in self.parts))

        return Partition([part for part in starmap(operator.sub, both) if part > 0])

    def __getitem__(self, ind):
        return self.parts[ind]

    def __iter__(self):
        for part in self.parts:
            yield part

    def __next__(self):
        if self.iter_index > self.end:
            raise StopIteration

        else:
            self.iter_index += 1
            return self.parts[self.iter_index]

    def __hash__(self):
        return hash(self.parts)

    def index(self, val, last=False):
        """ Index finder that uses optimized binary search because of the reverse-sorted nature of partitions."""
        left = 0
        right = len(self) - 1
        found_ind = -1

        while right >= left:
            middle = (right + left) >> 1

            if self[middle] == val:
                found_ind = middle
                break

            if self[middle] > val:
                left = middle + 1

            else:
                right = middle - 1

        if found_ind < 0:
            return -1

        if last:
            left = found_ind
            while right >= left:
                # found the last one
                try:
                    if self[found_ind + 1] < val:
                        return found_ind

                except IndexError:  # implies the current index was the last one.
                    return found_ind

                middle = (right + left) >> 1

                if self[middle] == val:
                    found_ind = middle
                    left = middle + 1

                else:
                    right = middle - 1

        else:
            right = found_ind
            while right >= left:
                # found the first one
                if found_ind == 0:  # implies the current index was the first one.
                    return found_ind

                if self[found_ind - 1] > val:
                    return found_ind

                middle = (right + left) >> 1

                if self[middle] == val:
                    found_ind = middle
                    right = middle + 1
                    continue

                else:
                    left = middle - 1

        return found_ind

    @staticmethod
    def reverse_insort(a, x, lo=0, hi=None):
        """ Reverse insort version of the bisect.insort method. """
        if lo < 0:
            raise ValueError("lo must be non-negative")
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x > a[mid]:
                hi = mid
            else:
                lo = mid + 1
        a.insert(lo, x)

    @cached_property
    def is_stable(self):
        return not np.where(np.diff(self) > -2)[0].size

    @cached_property
    def ar_parts(self):
        ar_parts = {}
        ar_starts = set()
        for ind, part in enumerate(self):
            if part in ar_starts:
                continue

            end_index = max(
                Partition(self[ind + 1 :]).index(part, last=True),
                Partition(self[ind + 1 :]).index(part - 1, last=True),
            )

            if end_index >= 0:
                ar_parts[ind] = self[ind : ind + end_index + 2]
                ar_starts.add(part)

        return ar_parts

    @cached_property
    def box_size(self):
        if not self or self.is_stable:
            return 0

        if len(self) == 1:
            return self[0]

        return (
            reduce(operator.mul, starmap(lambda x, y: x - y - 1, list(zip(self.parts, self.parts[1:]))))
            * self.parts[-1]
        )

    @cached_property
    def conjugate(self):
        """ https://en.wikipedia.org/wiki/Partition_(number_theory)#Conjugate_and_self-conjugate_partitions """
        new_conj = []
        parts_copy = list(self.parts)

        while parts_copy:
            new_conj.append(len(parts_copy))
            parts_copy = [part - 1 for part in parts_copy if part != 1]

        return Partition(new_conj)

    @cached_property
    def matrix(self):
        """ Simply returns a numpy representation of the matrix in a N x N grid where n is the
        sum of parts in the given partition instance. """
        grid = np.zeros((self.sum_of_parts, self.sum_of_parts))

        for ind, part in enumerate(self.parts):
            grid[ind][:part] = 1

        return grid

    def fit_matrix(self, n):

        """ Returns a numpy representation of the matrix in a N x N grid where n is provided. If n is less than
        the largest part of n, then the partition is fitted as tightly as possible. """
        n = max(max(self), n) if self else n
        grid = np.zeros((n, n))

        for ind, part in enumerate(self.parts):
            grid[ind][:part] = 1

        return grid

    @cached_property
    def sum_of_parts(self):
        """ Returns the sum of parts of the partition instance. """
        return sum(self.parts)

    @cached_property
    def durfee(self):
        """ https://en.wikipedia.org/wiki/Durfee_square """
        if not self.parts:
            return 0

        i = 0
        for ind, part in enumerate(self.parts):
            if part < ind + 1:
                break
            i += 1

        return i

    @cached_property
    def rank(self):
        """ https://en.wikipedia.org/wiki/Rank_of_a_partition """
        if not self.parts:
            return 0

        return self.parts[0] - len(self.parts)

    @cached_property
    def crank(self):
        """ https://en.wikipedia.org/wiki/Crank_of_a_partition """
        if not self.parts:
            return 0

        largest = self.parts[0]
        num_ones = self.parts.count(1)

        if not num_ones:
            return largest

        u = len([part for part in self.parts if part > num_ones])

        return u - num_ones

    @cached_property
    def rp(self):
        """ https://arxiv.org/pdf/1409.2192.pdf (Denoted as 'r sub p' in the paper). """
        return (-np.diff(self).clip(-2, 0).sum() // 2) + 1

    @cached_property
    def is_almost_rectangular(self):
        """ https://arxiv.org/pdf/1409.2192.pdf """
        if len(self) < 2:
            return False

        return (self[0] - self[-1]) <= 1

    @cached_property
    def hook_length(self):
        """ https://en.wikipedia.org/wiki/Young_tableau#Arm_and_leg_length """
        return self[0] + len(self) + 1 if self else 0

    @cached_property
    def num_corners(self):
        """ https://en.wikipedia.org/wiki/Hook_length_formula#Probabilistic_proof_using_the_hook_walk """
        matrix = self.matrix
        corner_count = 0

        for i, row in enumerate(matrix):
            non_zeros = np.nonzero(row)[0]
            if not non_zeros.any():
                break  # breaking is allowed by definition of partitions (they dont have "gaps")

            last_non_zero_index = non_zeros[-1]

            try:
                # check below (to the right is guaranteeed to be empty by construction)
                if not matrix[i + 1][last_non_zero_index]:
                    corner_count += 1

            except IndexError:  # must have been last row (which is a corner by definition)
                corner_count += 1

        return corner_count

    @cached_property
    def oblak(self):
        """ https://arxiv.org/pdf/1409.2192.pdf
        Returns the resulting partition after performing the partition Recursive Process as specified in the paper. """

        oblak = []
        partition = Partition(self[::])

        ar_parts = partition.ar_parts

        while ar_parts:
            max_u_chain_start_index = max(ar_parts, key=lambda x: 2 * x + sum(ar_parts[x]))
            max_u_chain_value = 2 * max_u_chain_start_index + sum(ar_parts[max_u_chain_start_index])

            if partition[0] >= max_u_chain_value:
                oblak.append(partition[0])
                partition = Partition(partition[1:])

            else:
                partition = Partition(
                    tuple([part - 2 for part in partition[:max_u_chain_start_index]])
                    + partition[max_u_chain_start_index + len(ar_parts[max_u_chain_start_index]) :]
                )

                oblak.append(max_u_chain_value)

            ar_parts = partition.ar_parts

        for part in partition:
            self.reverse_insort(oblak, part)

        return Partition(oblak)

    @cached_property
    def _next_oblak_step(self):
        """ This is a property that should only really be used for testing purposes. The only reason this
        exists is for neural network training purposes. Returns both the value and next partition that are
        a result of one iteration of the Oblak Recrusive Process. """

        oblak_value = 0
        partition = Partition(self[::])

        ar_parts = partition.ar_parts

        if ar_parts:
            max_u_chain_start_index = max(ar_parts, key=lambda x: 2 * x + sum(ar_parts[x]))
            max_u_chain_value = 2 * max_u_chain_start_index + sum(ar_parts[max_u_chain_start_index])

            if partition[0] >= max_u_chain_value:
                oblak_value = partition[0]
                partition = Partition(partition[1:])

            else:
                partition = Partition(
                    tuple([part - 2 for part in partition[:max_u_chain_start_index]])
                    + partition[max_u_chain_start_index + len(ar_parts[max_u_chain_start_index]) :]
                )

                oblak_value = max_u_chain_value
        return oblak_value, partition

