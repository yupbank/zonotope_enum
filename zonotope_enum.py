"""
2d zonotope vertex enumeration
"""

import numpy as np


def _angle(y_coor, x_coor):
    """
    Get 0-360 degree of a vector
    """
    return (np.arctan2(y_coor, x_coor)/np.pi*180 + 360) % 360


def _enum(gs):
    """
    Enumerate single quadrant sorted generators into vertices.
    """
    return np.cumsum(np.concatenate([gs, -gs]), axis=0)


def cluster(gs):
    """
    Cluster and sort generators into a single quadrant in counter-clockwise order.
    """
    degrees = _angle(gs[:, 1], gs[:, 0])
    orders = np.argsort(degrees)
    prev_index = 0
    for i in range(90, 361, 90):
        new_index = np.searchsorted(degrees, i, sorter=orders)
        if prev_index != new_index:
            yield gs[orders[prev_index:new_index]]
        prev_index = new_index


def angle(a, b):
    """
    Compute the directed angle of two vectors
    """
    c = b - a
    return _angle(c[1], c[0])


def merge_vertex(v, w):
    """
    Merge two sorted vertices lists.
    """
    i, j = 0, 0
    len_v, len_w = v.shape[0], w.shape[0]
    total = 2*(v.mean(axis=0) + w.mean(axis=0))
    res = []
    while True:
        res.append(v[i]+w[j])
        if angle(v[i % len_v], v[(i+1) % len_v]) < angle(w[j % len_w], w[(j+1) % len_w]):
            i = i+1
        else:
            j = j+1
        if i+j > (len_v+len_w)/2 - 1:
          break
    one_side = np.array(res)
    other_side = total - one_side
    return np.concatenate([one_side, other_side])


def roll(vs):
    """
    Re-arrange the counter-clockwise sorted vertices list starting with the least y-axis
    """
    return np.roll(vs, -np.argmin(vs[:, 1]), axis=0)


def enum(gs):
    """
    Enumerate vertices of zonotope based on generators
    """
    from functools import reduce
    v = [roll(_enum(i)) for i in cluster(gs)]
    return np.around(reduce(lambda x, y: roll(merge_vertex(x, y)), v), 10)
