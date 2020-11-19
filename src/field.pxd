# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.data_struct cimport *

cdef void advance_b_field(real4d, real4d, ints[6], double,double) nogil
cdef void advance_e_field(real4d, real4d, real4d, ints[6], double,double) nogil
