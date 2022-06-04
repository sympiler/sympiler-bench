__author__ = 'Kazem'

import pandas as pd
import sys
import getopt
from graph_utils import select_plotter, preprocess_csv, Index, c_labs, preprocess_text_file, global_dict
import glob
from stacked_graph_utils import plot_stacked_bar_for_files
#from profiling_analysis import *
#plot_3d_plot_profiling, plot_memory_cycle_locality, \
#    plot_redundant_comp, plot_redundant_versus_speedup, plot_locality_versus_speedup


PROFILING = False
import numpy as np



def poly_kernel(x, z, degree, intercept):
    return np.power(np.matmul(x, z.T) + intercept, degree)


def gaussian_kernel(x, z, sigma):
    n = x.shape[0]
    m = z.shape[0]
    xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
    zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))
    return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) / (2 * sigma ** 2))


def linear_kernel(x, z):
    return np.matmul(x, z.T)

def main(argv):
    output_path = ''
    input_path1 = ''
    input_dir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:d:o:", ["help", "inputPath=", "directory=", "outputPath="])
    except getopt.GetoptError:
        print('graph_gen.py -i <input csv> -d <input directory of csv files> -o <output directory to store graphs> ')
        sys.exit(-1)
    for opt, arg in opts:
        if opt == '-h':
            print('graph_gen.py -i <input csv> -d <input directory of csv files> -o <output directory to store graphs>')
        elif opt in ("-i", "--inputPath"):
            input_path1 = arg
        elif opt in ("-d", "--directory"):
            input_dir = arg.strip()
        elif opt in ("-o", "--outputPath"):
            output_path = arg

    if input_dir != '':
        files = [f for f in glob.glob(input_dir + "/*.csv", recursive=False)]
        if not PROFILING:
            plot_stacked_bar_for_files(files)
        sys.exit(0)

    if input_path1 != '':
        lib_file = False
        global_dict["ERRORS"] = preprocess_text_file(input_path1)
        in_csv_log = pd.read_csv(input_path1)
        lab1 = c_labs[Index.MAT_NAME.value]  # key label
        sorted_dic = preprocess_csv(in_csv_log, lab1, lab1)
        if "mkl" in input_path1:
            lib_file = True
        select_plotter(sorted_dic, output_path, lib_file)
        sys.exit(0)



if __name__ == "__main__":
    main(sys.argv[1:])


