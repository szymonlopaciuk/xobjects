# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

from xobjects.array import *


def np_f_strides(shape):
    return np.asfortranarray(np.zeros(shape)).strides


def np_c_strides(shape):
    return np.zeros(shape).strides


def test_strides():
    shape = (2, 3, 5, 7)

    c_strides = (3 * 5 * 7 * 8, 5 * 7 * 8, 7 * 8, 8)
    assert np_c_strides(shape) == c_strides
    assert get_strides(shape, (0, 1, 2, 3), 8) == c_strides

    f_strides = (8, 2 * 8, 2 * 3 * 8, 2 * 3 * 5 * 8)
    assert np_f_strides(shape) == f_strides
    assert get_strides(shape, (3, 2, 1, 0), 8) == f_strides


def test_iter_index():
    def check_ordering(shape, order, itemsize):
        order = mk_order(order, shape)
        strides = get_strides(shape, order, itemsize)
        old = -itemsize
        for idx in iter_index(shape, order):
            off = get_offset(idx, strides)
            assert off - old == itemsize
            old = off

    check_ordering((2, 3, 4), "C", 8)
    check_ordering((2, 3, 4), "F", 3)
    check_ordering((2, 3, 4), (2, 0, 1), 3)
