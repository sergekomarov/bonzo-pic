# -*- coding: utf-8 -*-

from bnz.defs cimport *

cdef void advance_bfld(real4d, real4d, int[6], real,real) nogil
cdef void advance_efld(real4d, real4d, real4d, int[6], real,real) nogil
