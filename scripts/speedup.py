__author__ = 'Kazem'

import pandas as pd
import sys
import getopt
import matplotlib.pyplot as plt
from os import path
from header_info import *

PROFILING = False


def plot(m0, m1, m2, m3, m4, m5, tools, y_label, f_out):
    fig = plt.figure(figsize=(80, 30))
    ax1 = fig.add_subplot(111)
    #title = "Symbolic + Factorization + Solve Time"
    #ax1.set_title(title, fontsize=100)
    # ax1.text(100, 7500, r'Common Memory Access Percentage = 2%', fontsize=50, fontweight='bold')
    # max_y = np.amax(data_array)+0.5

    ax1.plot(m0, label=tools[0], color='r')
    ax1.plot(m1, label=tools[1], color='k')
    ax1.plot(m2, label=tools[2], color='y')
    ax1.plot(m3, label=tools[3], color='g')
    ax1.plot(m4, label=tools[4], color='b')
    ax1.plot(m5, label=tools[5], color='m')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(100)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(100)

    plt.legend(loc='best', fontsize=100, ncol=1, frameon=False, prop={'size': 100})
    plt.xlabel("Matrix IDs", fontsize=100, weight="bold")
    plt.ylabel(y_label, fontsize=100, weight="bold")
    plt.show()
    fig.savefig(f_out+".pdf", bbox_inches='tight')

def get_data(input_path1):
    in_csv_log1 = pd.read_csv(input_path1)
    sym_time1 = in_csv_log1[SYMTIME].values
    fact_time1 = in_csv_log1[FACTIME].values + in_csv_log1[SOLVTIME].values
    tot_time1 = fact_time1 + sym_time1
    flops1 = in_csv_log1[FLOPS].values
    tool_name = in_csv_log1[TOOLN].values[0]
    return sym_time1, fact_time1, tot_time1, flops1, tool_name

def main(argv):
    output_path = ''
    input_path1 = ''
    input_path2 = ''
    input_dir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:j:k:d:o:", ["help", "inputPath1=", "inputPath2=", "directory=", "outputPath="])
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

    sym_time0, fact_time0, tot_time0, flops0, t0 = get_data(path.join(input_dir, "cholmod_parallel.csv"))
    sym_time1, fact_time1, tot_time1, flops1, t1 = get_data(path.join(input_dir, "mkl_pardiso_parallel.csv"))
    sym_time2, fact_time2, tot_time2, flops2, t2 = get_data(path.join(input_dir, "sympiler_parallel.csv"))
    sym_time3, fact_time3, tot_time3, flops3, t3 = get_data(path.join(input_dir, "sympiler_parallel_metis.csv"))
    sym_time4, fact_time4, tot_time4, flops4, t4 = get_data(path.join(input_dir, "sympiler_serial.csv"))
    sym_time5, fact_time5, tot_time5, flops5, t5 = get_data(path.join(input_dir, "sympiler_serial_metis.csv"))

    tool_list = [t0, t1, t2, t3, t4, t5]

    plot(fact_time0, fact_time1, fact_time2, fact_time3, fact_time4, fact_time5, tool_list, "Fact + Solve Time(sec)",
         "fact_solve")
    plot(tot_time0, tot_time1, tot_time2, tot_time3, tot_time4, tot_time5, tool_list, "Total Time(sec)",
         "total")
    plot(flops0/fact_time0, flops1/fact_time1, flops2/fact_time2, flops3/fact_time3, flops4/fact_time4, flops5/fact_time5,
         tool_list, "GFLOPs / sec", "flops")


def plot_mac(m0, m2, m3, m4, m5, tools, y_label, f_out):
    fig = plt.figure(figsize=(80, 30))
    ax1 = fig.add_subplot(111)
    #title = "Symbolic + Factorization + Solve Time"
    #ax1.set_title(title, fontsize=100)
    # ax1.text(100, 7500, r'Common Memory Access Percentage = 2%', fontsize=50, fontweight='bold')
    # max_y = np.amax(data_array)+0.5

    ax1.plot(m0, label=tools[0], color='r')
    ax1.plot(m2, label=tools[1], color='y')
    ax1.plot(m3, label=tools[2], color='g')
    ax1.plot(m4, label=tools[3], color='b')
    ax1.plot(m5, label=tools[4], color='m')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(100)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(100)

    plt.legend(loc='best', fontsize=100, ncol=1, frameon=False, prop={'size': 100})
    plt.xlabel("Matrix IDs", fontsize=100, weight="bold")
    plt.ylabel(y_label, fontsize=100, weight="bold")
    plt.show()
    fig.savefig(f_out+".pdf", bbox_inches='tight')

def main_mac(argv):
    output_path = ''
    input_path1 = ''
    input_path2 = ''
    input_dir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:j:k:d:o:", ["help", "inputPath1=", "inputPath2=", "directory=", "outputPath="])
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

    sym_time0, fact_time0, tot_time0, flops0, t0 = get_data(path.join(input_dir, "cholmod_parallel.csv"))
    sym_time2, fact_time2, tot_time2, flops2, t2 = get_data(path.join(input_dir, "sympiler_parallel.csv"))
    sym_time3, fact_time3, tot_time3, flops3, t3 = get_data(path.join(input_dir, "sympiler_parallel_metis.csv"))
    sym_time4, fact_time4, tot_time4, flops4, t4 = get_data(path.join(input_dir, "sympiler_serial.csv"))
    sym_time5, fact_time5, tot_time5, flops5, t5 = get_data(path.join(input_dir, "sympiler_serial_metis.csv"))

    tool_list = [t0, t2, t3, t4, t5]

    plot_mac(fact_time0, fact_time2, fact_time3, fact_time4, fact_time5, tool_list, "Fact + Solve Time(sec)",
         "fact_solve")
    plot_mac(tot_time0, tot_time2, tot_time3, tot_time4, tot_time5, tool_list, "Total Time(sec)",
         "total")
    plot_mac(flops0/fact_time0, flops2/fact_time2, flops3/fact_time3, flops4/fact_time4, flops5/fact_time5,
         tool_list, "GFLOPs / sec", "flops")


if __name__ == "__main__":
    main_mac(sys.argv[1:])



