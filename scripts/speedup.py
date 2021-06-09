__author__ = 'Kazem'

import pandas as pd
import sys
import getopt
import matplotlib.pyplot as plt

from header_info import *

PROFILING = False

def main(argv):
    output_path = ''
    input_path1 = ''
    input_path2 = ''
    input_dir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:j:d:o:", ["help", "inputPath1=", "inputPath2=", "directory=", "outputPath="])
    except getopt.GetoptError:
        print('graph_gen.py -i <input csv> -d <input directory of csv files> -o <output directory to store graphs> ')
        sys.exit(-1)
    for opt, arg in opts:
        if opt == '-h':
            print('graph_gen.py -i <input csv> -d <input directory of csv files> -o <output directory to store graphs>')
        elif opt in ("-i", "--inputPath1"):
            input_path1 = arg
        elif opt in ("-j", "--inputPath2"):
            input_path2 = arg
        elif opt in ("-d", "--directory"):
            input_dir = arg.strip()
        elif opt in ("-o", "--outputPath"):
            output_path = arg

    in_csv_log1 = pd.read_csv(input_path1)
    in_csv_log2 = pd.read_csv(input_path2)
    sym_time1 = in_csv_log1[SYMTIME].values
    sym_time2 = in_csv_log2[SYMTIME].values
    fact_time1 = in_csv_log1[FACTIME].values + in_csv_log1[SOLVTIME].values
    fact_time2 = in_csv_log2[FACTIME].values + in_csv_log2[SOLVTIME].values
    fact_time1 += sym_time1
    fact_time2 += sym_time2
    fig = plt.figure(figsize=(80, 30))
    ax1 = fig.add_subplot(111)
    title = "Symbolic + Factorization + Solve Time"
    ax1.set_title(title, fontsize=100)
    # ax1.text(100, 7500, r'Common Memory Access Percentage = 2%', fontsize=50, fontweight='bold')
    # max_y = np.amax(data_array)+0.5

    ax1.plot(fact_time1, label="MKL", color='r')
    ax1.plot(fact_time2, label="Sympiler", color='k')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(100)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(100)

    plt.legend(loc='best', fontsize=100, ncol=1, frameon=False, prop={'size': 100})
    plt.xlabel("Matrix IDs", fontsize=100, weight="bold")
    plt.ylabel("Time (sec)", fontsize=100, weight="bold")
    plt.show()
    fig.savefig("speedup.pdf", bbox_inches='tight')



if __name__ == "__main__":
    main(sys.argv[1:])



