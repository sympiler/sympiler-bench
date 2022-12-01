import scipy as sp
import scipy.sparse as sps
import sys

def make_lower(file_in, file_out):
    g_mat  = sp.io.mmread(file_in)
    g_mat_lower = sps.tril(g_mat)
    sp.io.mmwrite(file_out, g_mat_lower, symmetry='symmetric')


if __name__ == "__main__":
    make_lower(sys.argv[1], sys.argv[2])

