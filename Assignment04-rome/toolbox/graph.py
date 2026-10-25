"""The match graph: UnionFind, retrieval pairs, pair checks with visual words, component helpers and the
maximum spanning tree."""

import time

import numpy as np
import pandas as pd

from .data import _GROUPS, label_name, pretty
from .geometry import ransac_fundamental_cv


class UnionFind:
    """Items 0 .. n-1 in groups. Each group is a small tree, and its root names the group.

    find(i): follow parents up to the root; on the way back every visited item is hung straight under the root
             (path compression), so the next find is quick.
    union(i, j): hang the root of the smaller group under the root of the larger (union by size).
             Returns True if two groups were merged, False if i and j were already in one group.
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i):
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:                      # path compression
            nxt = self.parent[i]
            self.parent[i] = root
            i = nxt
        return root

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return False
        if self.size[ri] < self.size[rj]:
            ri, rj = rj, ri
        self.parent[rj] = ri                               # the smaller group goes under the larger
        self.size[ri] += self.size[rj]
        return True


def neighbour_pairs(idx):
    """Pairs (i, j), i < j, from a neighbour table idx (N,k): photo i with each of its neighbours. Sorted, unique."""
    idx = np.asarray(idx, dtype=np.int64)
    i = np.repeat(np.arange(len(idx)), idx.shape[1])
    j = idx.ravel()
    pairs = np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1)
    return np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)


def retrieval_precision(idx, labels):
    """Precision at k for k = 1 .. idx.shape[1]: the share of the first k neighbours that show the query's landmark,
    averaged over all queries that are not strays. Returns an array (k,)."""
    labels = np.asarray(labels)
    q = labels >= 0
    hit = labels[np.asarray(idx)[q]] == labels[q][:, None]
    return np.cumsum(hit, axis=1).mean(axis=0) / np.arange(1, hit.shape[1] + 1)


def _unique_words(rome):
    """For each photo: its words that occur exactly once in it (sorted) and their keypoint numbers. Cached."""
    if "_uniq" not in rome:
        off, w = rome["offsets"], rome["words"]
        uniq = []
        for i in range(len(off) - 1):
            vals, first, cnt = np.unique(w[off[i]:off[i + 1]], return_index=True, return_counts=True)
            uniq.append((vals[cnt == 1], first[cnt == 1] + off[i]))
        rome["_uniq"] = uniq
    return rome["_uniq"]


def word_matches(rome, i, j):
    """Tentative matches between photos i and j: keypoints whose visual word occurs exactly once in each photo.
    Returns (x1 (M,2), x2 (M,2)) pixels."""
    uniq = _unique_words(rome)
    _, a, b = np.intersect1d(uniq[i][0], uniq[j][0], assume_unique=True, return_indices=True)
    return rome["xy"][uniq[i][1][a]], rome["xy"][uniq[j][1][b]]


def verify_pairs(pairs, rome, min_inliers=20, thresh=2.0, iters=1000, verbose=True):
    """Check proposed pairs with geometry: word matches, then RANSAC on F (OpenCV, for speed).

    pairs (M,2). A pair is real when at least min_inliers matches fit one F within thresh px.
    Returns (n_inl (M,) int64 inlier counts, ok (M,) bool). Pairs with fewer than min_inliers word matches are not
    even tried (n_inl = 0). With verbose, long runs print a progress line every quarter.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    n_inl = np.zeros(len(pairs), dtype=np.int64)
    t0, step = time.time(), max(len(pairs) // 4, 1)
    for k, (i, j) in enumerate(pairs.tolist()):
        if verbose and len(pairs) > 10000 and k and k % step == 0:
            print(f"  checked {k:,} of {len(pairs):,} pairs, {np.sum(n_inl[:k] >= min_inliers):,} real so far "
                  f"({time.time() - t0:.0f} s)")
        x1, x2 = word_matches(rome, i, j)
        if len(x1) < min_inliers:
            continue
        _, inl = ransac_fundamental_cv(x1, x2, thresh=thresh, iters=iters)
        n_inl[k] = inl.sum()
    return n_inl, n_inl >= min_inliers


def component_purity(comp, labels, n_components=10, rome=None):
    """A table of the biggest components: size, the landmark most of its photos show, purity (the share of photos
    showing that landmark) and strays; then, for every landmark, the share of its photos in its biggest component.
    rome: the crate the labels come from, for the landmark names (default: the crate loaded last)."""
    comp, labels = np.asarray(comp), np.asarray(labels)
    groups = _GROUPS if rome is None else rome["groups"]
    rows = []
    for c in range(min(n_components, comp.max() + 1)):
        lab = labels[comp == c]
        lm = lab[lab >= 0]
        major = np.bincount(lm).argmax() if len(lm) else -1
        rows.append(dict(component=c, photos=len(lab), landmark=label_name(major, rome),
                         purity=f"{np.mean(lab == major):.0%}", strays=int(np.sum(lab < 0))))
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    rows = []
    for g in range(len(groups)):
        cs = comp[labels == g]
        if len(cs):
            best = np.bincount(cs).argmax()
            rows.append(dict(landmark=label_name(g, rome), photos=len(cs), main_component=int(best),
                             share=f"{np.mean(cs == best):.0%}"))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    alone = np.bincount(comp)[comp] == 1
    print(f"\nphotos left alone: {alone.sum()}, of which {np.sum(alone & (labels < 0))} are strays; "
          f"strays inside a component of 2 or more: {np.sum(~alone & (labels < 0))}")


def photos_of(rome, name):
    """The photos whose hidden landmark name contains name (e.g. "Trevi"), as a sorted int64 array. For pictures and
    examples only: your functions never use the hidden labels."""
    hits = [g for g, n in enumerate(rome["groups"]) if name.lower().replace(" ", "_") in n.lower()]
    if not hits:
        raise ValueError(f"no landmark called {name!r}; the crate has {', '.join(map(pretty, rome['groups']))}")
    return np.flatnonzero(np.isin(rome["labels"], hits)).astype(np.int64)


def component_of_landmark(comp, rome, name):
    """The component that holds most photos of the landmark whose name contains name (e.g. "Trevi")."""
    hits = [g for g, n in enumerate(rome["groups"]) if name.lower().replace(" ", "_") in n.lower()]
    if not hits:
        raise ValueError(f"no landmark called {name!r}; the crate has {', '.join(map(pretty, rome['groups']))}")
    cs = np.asarray(comp)[np.isin(rome["labels"], hits)]
    return int(np.bincount(cs).argmax())


def component_graph(comp, c, edges, weights):
    """The photos of component c and the edges among them, renumbered 0 .. n-1.
    Returns (members (n,) global photo numbers, edges (E,2) local numbers, weights (E,))."""
    comp = np.asarray(comp)
    members = np.flatnonzero(comp == c)
    local = -np.ones(len(comp), dtype=np.int64)
    local[members] = np.arange(len(members))
    edges, weights = np.asarray(edges, np.int64).reshape(-1, 2), np.asarray(weights)
    inside = (comp[edges[:, 0]] == c) & (comp[edges[:, 1]] == c)
    return members, local[edges[inside]], weights[inside]


def maximum_spanning_tree(n, edges, weights):
    """The skeleton's links: the strongest edges that still connect every photo of a component, with no loops.

    Kruskal's algorithm: visit the edges from heaviest to lightest and keep one only when it joins two separate
    groups (UnionFind.union returns True exactly then); an edge between two photos already joined would close a loop.
    One tree per connected component (a spanning forest), so keep.sum() == n - (number of components); a repeated pair
    keeps only its heaviest copy, and a self loop is never kept. Equal weights keep the edge listed first.

    n: photos 0 .. n-1; edges (E,2) ints; weights (E,) numbers (here inlier counts, bigger = stronger link); any
    array-like. Returns keep (E,) bool. With no repeated pairs, the same tree weight as
    scipy.sparse.csgraph.minimum_spanning_tree on -weights (scipy would add up the weights of a repeated pair).
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    uf = UnionFind(n)
    keep = np.zeros(len(edges), dtype=bool)
    for e in np.argsort(-weights, kind="stable"):            # heaviest first; ties in the order given
        keep[e] = uf.union(int(edges[e, 0]), int(edges[e, 1]))
    return keep


def skeleton_nodes(n, tree_edges, size=100):
    """The skeleton: peel the leaves off the spanning tree, round after round, while more than size photos remain
    (a round that would leave fewer than size is not done). Leaves hang off one strong link, so they can be placed
    later. Returns the sorted local numbers of the photos that remain."""
    tree_edges = np.asarray(tree_edges, np.int64).reshape(-1, 2)
    alive = np.zeros(n, bool)
    alive[np.unique(tree_edges)] = True
    while alive.sum() > size:
        live = tree_edges[alive[tree_edges[:, 0]] & alive[tree_edges[:, 1]]]
        deg = np.bincount(live.ravel(), minlength=n)
        leaves = alive & (deg <= 1)
        if not leaves.any() or alive.sum() - leaves.sum() < size:
            break
        alive &= ~leaves
    return np.flatnonzero(alive)
