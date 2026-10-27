# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dependency-free port of verl's multiway KK, with original tie-breaking.

Source: verl/utils/seqlen_balancing.py at
f2923cbf154731918b1e36db8c567433f349fede. Removed repr/docs/framework imports;
no claim of a full verl framework benchmark.
"""

import heapq

VERL_REVISION = "f2923cbf154731918b1e36db8c567433f349fede"


def karmarkar_karp(values, k, equal_size=True):
    if k < 1 or len(values) < k or (equal_size and len(values) % k):
        raise ValueError("KK needs nonempty partitions and divisible equal-size inputs")

    class Set:
        def __init__(self):
            self.sum, self.items = 0, []

        def add(self, idx, val):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.add(idx, val)

        def __lt__(self, other):
            return (self.sum, len(self.items), self.items) < (
                other.sum, len(other.items), other.items)

    class State:
        def __init__(self, items):
            self.sets = [Set() for _ in range(k)]
            for i, (idx, val) in enumerate(items):
                self.sets[i].add(idx, val)
            self.sets.sort(reverse=True)

        def merge(self, other):
            for i in range(k):
                self.sets[i].merge(other.sets[k - 1 - i])
            self.sets.sort(reverse=True)

        @property
        def spread(self):
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

    ordered = sorted((v, i) for i, v in enumerate(values))
    queue = []
    size = k if equal_size else 1
    for offset in range(0, len(ordered), size):
        heapq.heappush(queue, State([(i, v) for v, i in ordered[offset:offset + size]]))
    while len(queue) > 1:
        a, b = heapq.heappop(queue), heapq.heappop(queue)
        a.merge(b)
        heapq.heappush(queue, a)
    return [sorted(i for i, _ in part.items) for part in queue[0].sets]
