#!python
#cython: language_level=3
# from libc.string cimport  strlen #  this function takes a bytestring, not a python unicode string
import numpy as np
cimport numpy as np

np.import_array()


def edit_distance(str_a: str, str_b: str):
    # Levenshtein Distance between the two strings
    cdef int m = len(str_a)
    cdef int n = len(str_b)
    cdef np.ndarray d = np.zeros((m,n), dtype=int)
    cdef int i
    cdef int j

    for i in range(m):
        d[i, 0] = i

    for j in range(n):
        d[0, j] = j

    for j in range(n):
        for i in range(m):
            if str_a[i] == str_b[j]:
                sub_cost = 0
            else:
                sub_cost = 1

            d[i, j] = min(d[i-1, j] + 1,
                          d[i, j-1] + 1,
                          d[i-1, j-1] + sub_cost
                         )

    return d[m-1, n-1]


#def c_strlen(s: bytes):
#    return strlen(s)
