# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2023.                   #
# ########################################### #

import numpy as np
import xobjects as xo
from xobjects.test_helpers import requires_context

@requires_context('ContextPyopencl')
def test_array_masking():
    ctx = xo.ContextPyopencl()

    a = xo.Int64[:]([1, 2, 3, 4, 5, 6, 2], _context=ctx).to_nplike()
    b = xo.Int64[:]([6, 7, 1, 9, 4, 5, 2], _context=ctx).to_nplike()

    mask_a = a > b
    larger_a = a[mask_a]
    expected_a = ctx.nparray_to_context_array(np.array([3, 5, 6]))
    assert np.all(larger_a == expected_a)

    mask_b = a < b
    smaller_b = b[mask_b]
    expected_b = ctx.nparray_to_context_array(np.array([6, 7, 9]))
    assert np.all(smaller_b == expected_b)
