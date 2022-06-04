
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mtick

from graph_utils import *
plt.style.use('seaborn-whitegrid')
font0 = FontProperties()



SF_LABEL2R = "SF fused2 Red"
SF_LABEL2 = "SF fused2"
SF_LABEL4 = "SF fused4"
SF_LABEL6 = "SF fused6"
JD_WF2 = "Wavefront fused2"
JD_WF4 = "Wavefront fused4"
JD_LBC2 = "LBC fused2"
JD_LBC4 = "LBC fused4"
JD_DAGP = ""
PARSY = "LBC Non-fused"
SEQ = "Serial Non-fused"

SECs = " (sec)"
ANA = " Analysis"
ITR = " Iters"

NNZ = "A Nonzero"

ITR_MAX = 1020
MAX_VAL = 1e10


fig = plt.figure(figsize=(10, 4))



def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


def plot_scatter_chart_gs( all_data, legend_labels, x_axis_labels, y_axis_label,
                   out_file_name='', y_scale="linear", location=(0,0), colspan=1, legend_need=True):

    if len(all_data) <= 0:
        print("all_data invalid.")
        return -1
    n_groups = len(all_data)
    n_elements = len(all_data[0])
    for l in all_data:
        if len(l) != n_elements:
            print("all_data inconsistent.")
            return -1
    if len(legend_labels) != n_groups or len(x_axis_labels) != n_elements:
        print("legend_labels or x_axis_labels are invalid.")
        return -1

    width = 0.26
    max_y = np.amax(all_data)+2
    if legend_need:
        max_y = 4.75
    ind = (n_groups + 1) * width * np.arange(0,n_elements, (n_elements-1)/2)
    x_label = np.arange(1, n_elements + 1)
    y_label = np.arange(0, max_y, 5)
    font_axis = font0.copy()
    metric_label = 'solve time(sec)'

    #ax1 = fig.add_subplot(location)
    ax1 = plt.subplot2grid((1, 3), location, colspan=colspan)
    ax1.set_ylim([0, max_y])
    #ax1.set_xlim(1e6,1e8)
    ax1.set_yscale(y_scale)
    for i in range(n_groups):
        ax1.scatter(x_label, all_data[i, :], s=10, color=colors[i*3],
                    marker=markers[i], label=legend_labels[i])

    font_axis.set_weight(weights[6])  # ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
    if legend_need:
        ax1.set_ylabel(y_axis_label, fontsize=14)
    if location == (0, 1):
        ax1.set_xlabel("Number of NNZ", fontsize=14)
    #ax1.set_yticklabels()
    #ax1.set_yticks(y_label, minor=False)
    #ax1.set_yticks([0, 0.4, 0.8, 1.2, 1.6, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100 ])
    #ax1.axhline(y=[1], linestyle='--', color='gray')  #

    ax1.set_xticks(ind)
    x_axis_label_reduced = np.zeros(3)
    x_axis_label_reduced = np.array([x_axis_labels[0], x_axis_labels[int(n_elements/2)],  x_axis_labels[n_elements-1]])
    ax1.set_xticklabels(x_axis_label_reduced, fontsize=14)
    #ax1.xaxis.set_minor_formatter(mtick.FormatStrFormatter('%.2e'))
    #ax1.ticklabel_format(axis="x", style="sci")

    ax1.grid(False)
    ax1.tick_params(axis='y', which='both', left='off', right='off', color='w')
    ax1.tick_params(axis='x', which='both', left='off', right='off', color='w')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color('k')
    ax1.spines['bottom'].set_color('k')
    plt.yticks(Fontsize=14)
    plt.xlim(-width)
    plt.xticks(rotation=45)
    if legend_need:
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=14, ncol=3, loc='upper center', frameon=True, borderaxespad=1)

        #plt.legend(loc='upper left', fontsize=14, ncol=1,  frameon=True)
    #plt.show()
    dir_out = os.path.dirname(out_file_name)
    base_out = os.path.basename(out_file_name).split('.')[0]
    if os.path.exists(dir_out):
        fig.savefig(join(dir_out, base_out + ".pdf"), bbox_inches='tight')
    else:
        print("Output path is not valid, saving current dir!")
        fig.savefig(base_out + ".pdf", bbox_inches='tight')


def plot_gs_scatter(input_path1):
    lib_file = False
    global_dict["ERRORS"] = preprocess_text_file(input_path1)
    in_csv_log = pd.read_csv(input_path1)
    in_csv_log = in_csv_log.sort_values(by=[NNZ])
    lab1 = c_labs[Index.MAT_NAME.value]  # key label
    #sorted_dic = preprocess_csv(in_csv_log, lab1, lab1)

    NZ = in_csv_log[NNZ].values
    MATRICES = in_csv_log["Matrix Name"].values

    ### Baseline
    SERIAL = in_csv_log[SEQ + SECs].values

    ### ParSy
    PARSYV = in_csv_log[PARSY + SECs].values
    PARSYVI = in_csv_log[PARSY + ITR].values

    ### SF Fused
    SF2R = in_csv_log[SF_LABEL2R + SECs].values
    SF2 = in_csv_log[SF_LABEL2 + SECs].values
    SF4 = in_csv_log[SF_LABEL4 + SECs].values
    SF6 = in_csv_log[SF_LABEL6 + SECs].values

    SF2RI = in_csv_log[SF_LABEL2R + ITR].values
    SF2I = in_csv_log[SF_LABEL2 + ITR].values
    SF4I = in_csv_log[SF_LABEL4 + ITR].values
    SF6I = in_csv_log[SF_LABEL6 + ITR].values

    n = len(SF2)
    BEST_SF = np.zeros(n)
    BEST_SF_loop = np.zeros(n)
    BEST_WF_loop = np.zeros(n)
    BEST_LBC_loop = np.zeros(n)
    min_vals = np.zeros(4)
    best_k = np.zeros(4)
    for i in range(n):
        min_vals[0] = SF2R[i] if SF2RI[i] == PARSYVI[i] else MAX_VAL
        min_vals[1] = SF2[i] if SF2I[i] == PARSYVI[i] or abs(SF2I[i] - PARSYVI[i])==1 else MAX_VAL
        min_vals[2] = SF4[i] if SF4I[i] == PARSYVI[i] else MAX_VAL
        min_vals[3] = SF6[i] if SF6I[i] == PARSYVI[i] else MAX_VAL
        BEST_SF[i] = np.min(min_vals)
        if BEST_SF[i] == min_vals[0] or BEST_SF[i] == min_vals[1]:
            BEST_SF_loop[i] = 2
        if BEST_SF[i] == min_vals[2]:
            BEST_SF_loop[i] = 4
        if BEST_SF[i] == min_vals[3]:
            BEST_SF_loop[i] = 6

        if BEST_SF[i] == MAX_VAL:
            print( "===> ", i)
        num = np.where(BEST_SF[0] == min_vals)[0]
        best_k[num] += 1


    JD_WF2V = in_csv_log[JD_WF2 + SECs].values
    JD_WF4V = in_csv_log[JD_WF4 + SECs].values
    JD_LBC2V = in_csv_log[JD_LBC2 + SECs].values
    JD_LBC4V = in_csv_log[JD_LBC4 + SECs].values

    JD_WF2VI = in_csv_log[JD_WF2 + ITR].values
    JD_WF4I = in_csv_log[JD_WF4 + ITR].values
    JD_LBC2VI = in_csv_log[JD_LBC2 + ITR].values
    JD_LBC4VI = in_csv_log[JD_LBC4 + ITR].values

    BEST_JD = np.zeros(n)
    min_vals = np.zeros(4)
    for i in range(n):
        min_vals[0] = JD_WF2V[i] if JD_WF2VI[i] == PARSYVI[i] else MAX_VAL
        min_vals[1] = JD_WF4V[i] if JD_WF4I[i] == PARSYVI[i] else MAX_VAL
        min_vals[2] = JD_LBC2V[i] if JD_LBC2VI[i] == PARSYVI[i] else MAX_VAL
        min_vals[3] = JD_LBC4V[i] if JD_LBC4VI[i] == PARSYVI[i] else MAX_VAL
        BEST_JD[i] = np.min(min_vals)
        best_wf = np.min((min_vals[0], min_vals[1]))
        if best_wf == min_vals[0]:
            BEST_WF_loop[i] = 2
        if best_wf == min_vals[1]:
            BEST_WF_loop[i] = 4

        best_jd = np.min((min_vals[2], min_vals[3]))
        if best_jd == min_vals[2]:
            BEST_LBC_loop[i] = 2
        if best_jd == min_vals[3]:
            BEST_LBC_loop[i] = 4

        if BEST_JD[i] == MAX_VAL:
            print( "===> ", i)
        num = np.where(BEST_SF[0] == min_vals)[0]

    print("AVG Speedup over ParSy:", np.average(PARSYV/BEST_SF))
    print("Max Speedup over ParSy:", np.max(PARSYV / BEST_SF))
    print("Min Speedup over ParSy:", np.min(PARSYV / BEST_SF))
    print("AVG Speedup over JD:", np.average(BEST_JD / BEST_SF))
    print("Max Speedup over JD:", np.max(BEST_JD / BEST_SF))
    print("Min Speedup over JD:", np.min(BEST_JD / BEST_SF))

    sw = False
    if sw:
        filename = "GS_K_Fusion"

        half = 30
        data_array1 = np.zeros((3, half))
        # data_array[0, :] = SERIAL
        data_array1[0, :] = BEST_SF[0:half]
        data_array1[1, :] = PARSYV[0:half]
        data_array1[2, :] = BEST_JD[0:half]
        plot_scatter_chart_gs(data_array1, ["GS-Sparse-Fusion", "GS-ParSy", "GS-Joint-DAG"], NZ[0:half],
                              "Seconds", filename, "linear", (0, 0), 1, True)

        half1 = 63
        data_array1 = np.zeros((3, half1 - half))
        # data_array[0, :] = SERIAL
        data_array1[0, :] = BEST_SF[half:half1]
        data_array1[1, :] = PARSYV[half:half1]
        data_array1[2, :] = BEST_JD[half:half1]
        plot_scatter_chart_gs(data_array1, ["GS-Sparse-Fusion", "GS-ParSy", "GS-Joint-DAG"], NZ[half:half1],
                              "Seconds", filename, "linear", (0, 1), 1, False)

        data_array1 = np.zeros((3, n - half1))
        # data_array[0, :] = SERIAL
        data_array1[0, :] = BEST_SF[half1:n]
        data_array1[1, :] = PARSYV[half1:n]
        data_array1[2, :] = BEST_JD[half1:n]
        plot_scatter_chart_gs(data_array1, ["GS-Sparse-Fusion", "GS-ParSy", "GS-Joint-DAG"], NZ[half1:n],
                              "Seconds", filename, "linear", (0, 2), 1, False)

        data_array_loopno = np.zeros((3, n))

        # plt.legend(loc='upper left', fontsize=14, ncol=1, frameon=True)

    else:
        # data_array1 = np.zeros((3, n))
        # # data_array[0, :] = SERIAL
        # data_array1[0, :] = BEST_SF_loop[0:n]
        # data_array1[1, :] = BEST_WF_loop[0:n]
        # data_array1[2, :] = BEST_LBC_loop[0:n]
        # plot_scatter_chart_gs(data_array1, ["GS-Sparse-Fusion", "GS-ParSy", "GS-Joint-DAG"], NZ[0:n],
        #                       "Number of loops", filename, "linear", (0, 2), 1, False)

        print("AVG Speedup over two loops:", np.average(SF2 / BEST_SF))
        print("min Speedup over two loops:", np.min(SF2 / BEST_SF))
        print("max Speedup over two loops:", np.max(SF2 / BEST_SF))
        filename = "GS_K_Fusion_twomore"
        legend =  ["GS-Sparse-Fusion \n(best of 2-6 loops)", "GS-Sparse-Fusion \n(only two loops)"]
        half = 30
        data_array2 = np.zeros((2, half))
        # data_array[0, :] = SERIAL
        data_array2[0, :] = BEST_SF[0:half]
        data_array2[1, :] = SF2[0:half]
        plot_scatter_chart_gs(data_array2, legend, NZ[0:half],
                              "Seconds", filename, "linear", (0, 0), 1, True)

        half1 = 63
        data_array2 = np.zeros((2, half1 - half))
        # data_array[0, :] = SERIAL
        data_array2[0, :] = BEST_SF[half:half1]
        data_array2[1, :] = SF2[half:half1]
        plot_scatter_chart_gs(data_array2, legend, NZ[half:half1],
                              "Seconds", filename, "linear", (0, 1), 1, False)

        data_array2 = np.zeros((2, n - half1))
        # data_array[0, :] = SERIAL
        data_array2[0, :] = BEST_SF[half1:n]
        data_array2[1, :] = SF2[half1:n]
        plot_scatter_chart_gs(data_array2,legend, NZ[half1:n],
                              "Seconds", filename, "linear", (0, 2), 1, False)









    plt.show()



if __name__ == "__main__":
    args = sys.argv[1:]
    plot_gs_scatter(args[0])

