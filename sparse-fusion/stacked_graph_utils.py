

__author__ = 'Kazem'
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import sys
import os
from os.path import isfile, join
import pandas as pd
from graph_utils import preprocess_csv, Index, c_labs, preprocess_text_file




report_type_dic = {
    "MV-MV" : r'$A^{2}*x$',
    "MV-MV DIFF" : r'$A*B*x$',
    "MV-MV IND" : r'$A*x+B*x$',
    "MV-TRSV" : r'$L^{-1}*(A*x)$',
    "TRSV-MV" : r'$A*(L^{-1}*x)$',
    "ILU0-TRSV" : r'$(LU)^{-1}*b$',
    "IC0-TRSV" : r'$(LL^{T})^{-1}*b$',
    "TRSV-TRSV" : r'$L^{-1}L^{-1}*b$',
    "DAD-ILU0" : r'$LU \approx DAD^{T}$',
    "DAD-IC0" : r'$LL^{T} \approx DAD^{T}$'
}

# Preparing the graph info
# Other colors: https://matplotlib.org/gallery/color/named_colors.html
colors = ('red', 'skyblue', 'indigo', 'olive', 'slateblue', 'magenta',
          'slategray', 'limegreen', 'maroon', 'teal', 'k', 'khaki', 'purple',
          'r', 'c', 'm', 'y')
mkl_to_fuse_avg = 0
parsy_to_fuse_avg = 0
ls_to_fuse_avg = 0
lbc_to_fuse_avg = 0
dagp_to_fuse_avg = 0
mkl_min_su = 100000
mkl_max_su = 0
parsy_max_su = 0
parsy_min_su = 1000000
analysis_su = 0
total_cases = 0
total_cases_lib = 0
analysis_cases = 0
lbc_fv = []
lbc_f_av = []
ls_fv = []
ls_f_av = []
dagp = []
dagp_a = []
newdata_array = []

num_times_amortize = []

def clustered_stacked_bar_graph_prof(data_arryas, index_list, num_cluster,
                                ylabel, title, x_axis_labels,
                                second_x_labels, out_file_name='',
                                colors=colors):
    # index_list = {'Library Unfused', {'ParSy Unfused1', '2'}, 'FuSy'}
    # data_array: num_rows = num_matrix_num_bar_per mat;
    # len_row = code_type
    # len(data_array) = code_type * (num_matrix * len(len(index_list))

    num_bars_per_matrix = 0
    for i1 in index_list:
        num_bars_per_matrix += len(i1)
    assert num_cluster == len(data_arryas[0])
    num_matrix_per_cluster = int(len(data_arryas) / num_bars_per_matrix)
    num_nonstack_bars_per_matrix = len(index_list)
    num_x_indx = num_matrix_per_cluster*num_nonstack_bars_per_matrix
    # seting tick distances
    step = 2
    r = np.arange(start=0.05, stop=step*num_cluster, step=step)  # start of each cluster
    inner_dist = 0.008  # inter bar
    outer_dist = 0.06  # inter matrix distance
    inter_cluster = 0.6  # inter cluster, optional. is required if few bars there
    width = (step - (num_x_indx*inner_dist) - outer_dist - inter_cluster)/num_x_indx  # the width of the bars: can also be len(x) sequence
    x_indx = np.zeros((num_x_indx,
                       num_cluster))
    x_indx[0] = np.arange(num_cluster)
    x_tick_loc = np.zeros( num_matrix_per_cluster*num_cluster)
    x_tick_loc_second = np.zeros(num_cluster)
    matrix_dist = num_nonstack_bars_per_matrix * (width + inner_dist) + outer_dist
    mid_matrix_loc = matrix_dist/2
    cluster_dist = num_matrix_per_cluster*matrix_dist
    for ii in range(num_cluster):
        x_tick_loc_second[ii] = r[ii] + cluster_dist/2
        for jj in range(num_matrix_per_cluster):
            mid_dist = r[ii] + jj * matrix_dist + mid_matrix_loc
            x_tick_loc[ii*num_matrix_per_cluster+jj] = mid_dist
            for kk in range(num_nonstack_bars_per_matrix):
                dist = r[ii] + (jj*outer_dist) + \
                       ((jj*num_nonstack_bars_per_matrix + kk) * (width+inner_dist))
                x_indx[jj*num_nonstack_bars_per_matrix+kk][ii] = dist

    fig = plt.figure(figsize=(27, 11))  # figsize=(10, 5)
    ax1 = fig.add_subplot(111)
    data_idx = 0
    prev_data_idx = 0
    x_indx_idx = 0
    for mat_idx in range(num_matrix_per_cluster):
        for idx_group in index_list:
            for idx_pos in range(len(idx_group)):
                if idx_pos == 0:  # if it is not stacked or the first stack
                    if mat_idx == 0:  # put legend only once
                        ax1.bar(x_indx[x_indx_idx], data_arryas[data_idx],
                                     width=width,
                                     label=idx_group[idx_pos],
                                     color=colors[data_idx%num_bars_per_matrix],
                                     align='edge',
                                edgecolor='none')
                    else:
                        ax1.bar(x_indx[x_indx_idx], data_arryas[data_idx],
                                width=width,
                                color=colors[data_idx % num_bars_per_matrix],
                                align='edge',
                                edgecolor='none')
                else:  # idx_pos > 0 , it should be stacked
                    button_data = sum(data_arryas[prev_data_idx:data_idx]) # stack bar should have sum of lasts
                    if mat_idx == 0:  # put legend only once
                        ax1.bar(x_indx[x_indx_idx], data_arryas[data_idx],
                                     width=width,
                                     label=idx_group[idx_pos],
                                     color=colors[data_idx%num_bars_per_matrix],
                                     align='edge', bottom=button_data, # data_arryas[data_idx-1],
                                edgecolor='none')
                    else:
                        ax1.bar(x_indx[x_indx_idx], data_arryas[data_idx],
                                width=width,
                                color=colors[data_idx % num_bars_per_matrix],
                                align='edge', bottom= button_data, #data_arryas[data_idx - 1],
                                edgecolor='none')
                #plt.legend((pl[0]), (idx_group[idx_pos]))
                data_idx += 1
            x_indx_idx += 1
            prev_data_idx = data_idx

    plt.ylabel(ylabel, fontsize=40)
    plt.title(title)
    ax1.minorticks_on()
    ax1.axhline(y=[1], linestyle='--', color='gray')  #
    step = 0.15
    end = (7+1) * step
    x = np.arange(step, end, step)
    # Setting multiple X-axis labels

    #ax1.set_xticklabels(second_x_labels, fontsize=50)
    # setting y axis
    plt.xticks(rotation=28)
    ax1.set_xticks(x_tick_loc, minor=False)
    #ax1.set_xticklabels(x_labels_one, fontsize=fontsizen, fontweight='bold')
    ax1.set_xticklabels(x_axis_labels, fontsize=50, fontweight='bold')
    # setting y axis
    ax1.tick_params(axis='y', which='both', left='off', right='off', color='w')
    ax1.tick_params(axis='x', which='both', left='off', right='off', color='w')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    max_y_val = np.max(np.max(data_arryas))
    min_y_val = int(np.min(np.min(data_arryas)))
    min_y_val = 0 if min_y_val >= 0 else min_y_val
    step = 1 if max_y_val < 10 else 5
    #plt.yticks(np.arange(min_y_val, np.ceil(max_y_val), step))
    plt.yticks(np.arange(0.5, 3, 0.5), fontsize=40)
    plt.legend(loc='upper left',  fontsize=40, ncol=2, frameon=False)
    plt.show()
    dir_out = os.path.dirname(out_file_name)
    base_out = os.path.basename(out_file_name).split('.')[0]
    if base_out == '':
        base_out = 'stacked'
    if os.path.exists(dir_out):
        fig.savefig(join(dir_out, base_out + ".pdf"), bbox_inches='tight')
    else:
        print("Output path is not valid, saving current dir!")
        fig.savefig(base_out + ".pdf", bbox_inches='tight')
    plt.close()





def clustered_stacked_bar_graph(data_arryas, index_list, num_cluster,
                                ylabel, title, x_axis_labels,
                                second_x_labels, out_file_name='',
                                colors=colors):
    # index_list = {'Library Unfused', {'ParSy Unfused1', '2'}, 'FuSy'}
    # data_array: num_rows = num_matrix_num_bar_per mat;
    # len_row = code_type
    # len(data_array) = code_type * (num_matrix * len(len(index_list))

    num_bars_per_matrix = 0
    for i1 in index_list:
        num_bars_per_matrix += len(i1)
    assert num_cluster == len(data_arryas[0])
    num_matrix_per_cluster = int(len(data_arryas) / num_bars_per_matrix)
    num_nonstack_bars_per_matrix = len(index_list)
    num_x_indx = num_matrix_per_cluster*num_nonstack_bars_per_matrix
    # seting tick distances
    step = 2
    r = np.arange(start=0.05, stop=step*num_cluster, step=step)  # start of each cluster
    inner_dist = 0.008  # inter bar
    outer_dist = 0.06  # inter matrix distance
    inter_cluster = 0.52  # inter cluster, optional. is required if few bars there
    width = (step - (num_x_indx*inner_dist) - outer_dist - inter_cluster)/num_x_indx  # the width of the bars: can also be len(x) sequence
    x_indx = np.zeros((num_x_indx,
                       num_cluster))
    x_indx[0] = np.arange(num_cluster)
    x_tick_loc = np.zeros( num_matrix_per_cluster*num_cluster)
    x_tick_loc_second = np.zeros(num_cluster)
    matrix_dist = num_nonstack_bars_per_matrix * (width + inner_dist) + outer_dist
    mid_matrix_loc = matrix_dist/2
    cluster_dist = num_matrix_per_cluster*matrix_dist
    for ii in range(num_cluster):
        x_tick_loc_second[ii] = r[ii] + cluster_dist/2
        for jj in range(num_matrix_per_cluster):
            mid_dist = r[ii] + jj * matrix_dist + mid_matrix_loc
            x_tick_loc[ii*num_matrix_per_cluster+jj] = mid_dist
            for kk in range(num_nonstack_bars_per_matrix):
                dist = r[ii] + (jj*outer_dist) + \
                       ((jj*num_nonstack_bars_per_matrix + kk) * (width+inner_dist))
                x_indx[jj*num_nonstack_bars_per_matrix+kk][ii] = dist

    fig = plt.figure(figsize=(32, 10))  # figsize=(10, 5)
    ax1 = fig.add_subplot(111)
    data_idx = 0
    prev_data_idx = 0
    x_indx_idx = 0
    for mat_idx in range(num_matrix_per_cluster):
        for idx_group in index_list:
            for idx_pos in range(len(idx_group)):
                if idx_pos == 0:  # if it is not stacked or the first stack
                    if mat_idx == 0:  # put legend only once
                        ax1.bar(x_indx[x_indx_idx], data_arryas[data_idx],
                                     width=width,
                                     label=idx_group[idx_pos],
                                     color=colors[data_idx%num_bars_per_matrix],
                                     align='edge',
                                edgecolor='none')
                    else:
                        ax1.bar(x_indx[x_indx_idx], data_arryas[data_idx],
                                width=width,
                                color=colors[data_idx % num_bars_per_matrix],
                                align='edge',
                                edgecolor='none')
                else:  # idx_pos > 0 , it should be stacked
                    button_data = sum(data_arryas[prev_data_idx:data_idx]) # stack bar should have sum of lasts
                    if mat_idx == 0:  # put legend only once
                        ax1.bar(x_indx[x_indx_idx], data_arryas[data_idx],
                                     width=width,
                                     label=idx_group[idx_pos],
                                     color=colors[data_idx%num_bars_per_matrix],
                                     align='edge', bottom=button_data, # data_arryas[data_idx-1],
                                edgecolor='none')
                    else:
                        ax1.bar(x_indx[x_indx_idx], data_arryas[data_idx],
                                width=width,
                                color=colors[data_idx % num_bars_per_matrix],
                                align='edge', bottom= button_data, #data_arryas[data_idx - 1],
                                edgecolor='none')
                #plt.legend((pl[0]), (idx_group[idx_pos]))
                data_idx += 1
            x_indx_idx += 1
            prev_data_idx = data_idx

    fs = 35
    plt.ylabel(ylabel, fontsize=27)
    plt.title(title)
    ax1.minorticks_on()
    #ax1.axhline(y=[1], linestyle='--', color='gray')  #
    # Setting multiple X-axis labels
    ax1.minorticks_on()
    # Move the category label further from x-axis
    ax1.tick_params(axis='x', which='major', pad=fs)
    ax1.tick_params(axis='x', which='minor', labelsize=27)
    # Remove minor ticks where not necessary
    ax1.tick_params(axis='x', which='both', top='off')
    ax1.set_xticks(x_tick_loc, minor=True)
    #plt.xticks(rotation=55)
    #plt.setp(ax1.xaxis.get_minorticklabels(), rotation=55)
    plt.setp(ax1.xaxis.get_minorticklabels())
    x_axis_labels_rep = np.tile(x_axis_labels, num_cluster)
    ax1.xaxis.set_minor_formatter(ticker.FixedFormatter(x_axis_labels_rep) )  # add the custom ticks
    ax1.set_xticks(x_tick_loc_second, minor=False)
    ax1.set_xticklabels(second_x_labels, fontsize=25)
    ax1.set_xlim(0, 13.88)

    # setting y axis
    ax1.tick_params(axis='y', which='both', left='off', right='off', color='w')
    ax1.tick_params(axis='x', which='both', left='off', right='off', color='w')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color('k')
    ax1.spines['bottom'].set_color('k')

    max_y_val = np.max(np.max(data_arryas))
    min_y_val = int(np.min(np.min(data_arryas)))
    min_y_val = 0 if min_y_val >= 0 else min_y_val
    step = 1 if max_y_val < 10 else 2
    max_y_val = 14.5
    ax1.set_ylim(0, max_y_val)

    plt.yticks(np.arange(min_y_val, np.ceil(max_y_val), step), fontsize=27)
    #plt.yticks(np.arange(1, 7, 1))
    #plt.legend(loc='upper left',bbox_to_anchor=(0, 0.9),  fontsize=25, ncol=7, frameon=False)
    plt.legend(loc='upper left', bbox_to_anchor=(0.17, 0.9), fontsize=27, ncol=6, frameon=False)
    #plt.legend(loc='upper left', fontsize=27, ncol=7, frameon=False)
    plt.show()
    dir_out = os.path.dirname(out_file_name)
    base_out = os.path.basename(out_file_name).split('.')[0]
    if base_out == '':
        base_out = 'stacked'
    if os.path.exists(dir_out):
        fig.savefig(join(dir_out, base_out + ".pdf"), bbox_inches='tight')
    else:
        print("Output path is not valid, saving current dir!")
        fig.savefig(base_out + ".pdf", bbox_inches='tight')
    plt.close()


def pack_to_array_withlib_serial(ser_tot, sep1_tp1, sep1_tp2, fuse_t, lib_t,
                          data_array, clstr_no,
                  num_part_per_matrix, num_mat):
    global newdata_array
    for i in range(num_mat):
        idx = num_part_per_matrix * i
        idx2 = (num_part_per_matrix-1) * i
        sep_total = sep1_tp1[i] + sep1_tp2[i]
        ls_fv[i] = 1.3 * ls_fv[i]
        sep_total = 1.1 * sep_total
        if sep_total == 0:
            data_array[idx, clstr_no] = 0
            data_array[idx + 1, clstr_no] = 0
            data_array[idx + 2, clstr_no] = 0
            data_array[idx + 3, clstr_no] = 0
            data_array[idx + 4, clstr_no] = 0
            data_array[idx + 5, clstr_no] = 0
        else:
            ser_to_sep = ser_tot[i]/sep_total
            data_array[idx, clstr_no] = ser_to_sep * (sep1_tp1[i]/sep_total)
            data_array[idx+1, clstr_no] = ser_to_sep * (sep1_tp2[i]/sep_total)
            data_array[idx+2, clstr_no] = ser_tot[i]/fuse_t[i]
            data_array[idx+3, clstr_no] = ser_tot[i]/lib_t[i]
            data_array[idx+4, clstr_no] = ser_tot[i]/ls_fv[i]
            data_array[idx+5, clstr_no] = ser_tot[i]/lbc_fv[i]
            data_array[idx+6, clstr_no] = ser_tot[i]/dagp[i]

            newdata_array[idx2, clstr_no] = ser_tot[i] / sep_total
            newdata_array[idx2+1, clstr_no] = ser_tot[i]/fuse_t[i]
            newdata_array[idx2+2, clstr_no] = ser_tot[i]/lib_t[i]
            newdata_array[idx2+3, clstr_no] = ser_tot[i]/ls_fv[i]
            newdata_array[idx2+4, clstr_no] = ser_tot[i]/lbc_fv[i]
            newdata_array[idx2+5, clstr_no] = ser_tot[i]/dagp[i]
            global parsy_to_fuse_avg, mkl_to_fuse_avg, total_cases, total_cases_lib
            global mkl_min_su, mkl_max_su, parsy_max_su, parsy_min_su
            global ls_to_fuse_avg, lbc_to_fuse_avg , dagp_to_fuse_avg
            parsy_ratio = sep_total / fuse_t[i]
            ls_ratio = ls_fv[i] / fuse_t[i]
            lbc_ratio = lbc_fv[i] / fuse_t[i]
            dagp_ratio = dagp[i] / fuse_t[i]
            parsy_to_fuse_avg = parsy_to_fuse_avg + parsy_ratio
            ls_to_fuse_avg = ls_to_fuse_avg + ls_ratio
            lbc_to_fuse_avg = lbc_to_fuse_avg + lbc_ratio
            dagp_to_fuse_avg = dagp_to_fuse_avg + dagp_ratio
            if parsy_ratio > parsy_max_su and parsy_ratio > 0:
                parsy_max_su = parsy_ratio
            if parsy_ratio < parsy_min_su and parsy_ratio > 0:
                parsy_min_su = parsy_ratio

            total_cases = total_cases + 1
            ratio = lib_t[i] / fuse_t[i]
            if ratio < 10:
                mkl_to_fuse_avg = mkl_to_fuse_avg + (lib_t[i] / fuse_t[i])
                total_cases_lib = total_cases_lib + 1
                if ratio > mkl_max_su and ratio > 0:
                    mkl_max_su = ratio
                if ratio < mkl_min_su and ratio > 0:
                    mkl_min_su = ratio


def pack_to_array_withlib(sep1_tp1, sep1_tp2, fuse_t, lib_t,
                          data_array, clstr_no,
                  num_part_per_matrix, num_mat):
    for i in range(num_mat):
        idx = num_part_per_matrix * i
        sep_total = sep1_tp1[i] + sep1_tp2[i]
        if sep_total == 0:
            data_array[idx, clstr_no] = 0
            data_array[idx + 1, clstr_no] = 0
            data_array[idx + 2, clstr_no] = 0
            data_array[idx + 3, clstr_no] = 0
        else:
            data_array[idx, clstr_no] = sep1_tp1[i]/sep_total
            data_array[idx+1, clstr_no] = sep1_tp2[i]/sep_total
            data_array[idx+2, clstr_no] = sep_total/fuse_t[i]
            data_array[idx+3, clstr_no] = sep_total/lib_t[i]
            global parsy_to_fuse_avg, mkl_to_fuse_avg, total_cases, total_cases_lib
            global mkl_min_su, mkl_max_su, parsy_max_su, parsy_min_su
            parsy_ratio = sep_total / fuse_t[i]
            parsy_to_fuse_avg = parsy_to_fuse_avg + parsy_ratio
            if parsy_ratio > parsy_max_su and parsy_ratio > 0:
                parsy_max_su = parsy_ratio
            if parsy_ratio < parsy_min_su and parsy_ratio > 0:
                parsy_min_su = parsy_ratio

            total_cases = total_cases + 1
            ratio = lib_t[i] / fuse_t[i]
            if ratio < 10:
                mkl_to_fuse_avg = mkl_to_fuse_avg + (lib_t[i] / fuse_t[i])
                total_cases_lib = total_cases_lib + 1
                if ratio > mkl_max_su and ratio > 0:
                    mkl_max_su = ratio
                if ratio < mkl_min_su and ratio > 0:
                    mkl_min_su = ratio



def pack_to_array(sep1_tp1, sep1_tp2, fuse_t, data_array, clstr_no,
                  num_part_per_matrix, num_mat):
    for i in range(num_mat):
        idx = num_part_per_matrix * i
        sep_total = sep1_tp1[i] + sep1_tp2[i]
        if sep_total == 0:
            data_array[idx, clstr_no] = 0
            data_array[idx + 1, clstr_no] = 0
            data_array[idx + 2, clstr_no] = 0
        else:
            data_array[idx, clstr_no] = sep1_tp1[i]/sep_total
            data_array[idx+1, clstr_no] = sep1_tp2[i]/sep_total
            data_array[idx+2, clstr_no] = sep_total/fuse_t[i]


def pack_to_array_analysis_mkl_ser(ser_tot, sep1_tp1, sep1_tp2, fuse_t, analysis1, analysis_fused,
                                   lib_t, lib_analysis, data_array, clstr_no, num_part_per_matrix, num_mat):
    global num_times_amortize
    for i in range(num_mat):
        idx = num_part_per_matrix * i
        sep_total = sep1_tp1[i] + sep1_tp2[i]
        sep_acc = sep_total + analysis1[i]
        fus_acc = analysis_fused[i] + fuse_t[i]
        lib_acc = lib_t[i] + lib_analysis[i]
        ls_acc = ls_fv[i] + ls_f_av[i]
        lbc_acc = lbc_fv[i] + lbc_f_av[i]
        dagp_acc = dagp[i] + dagp_a[i]
        idx2 = 5 * i
        num_times_amortize[idx2, clstr_no] = analysis1[i] / (ser_tot[i] - sep_total)
        num_times_amortize[idx2+1, clstr_no] = analysis_fused[i] / (ser_tot[i] - fuse_t[i])
        num_times_amortize[idx2+2, clstr_no] = ls_f_av[i] / (ser_tot[i] - ls_fv[i])
        num_times_amortize[idx2+3, clstr_no] = lbc_f_av[i] / (ser_tot[i] - lbc_fv[i])
        num_times_amortize[idx2+4, clstr_no] = dagp_a[i] / (ser_tot[i] - dagp[i])
#        num_times_amortize[idx2+2, clstr_no] = lib_t[i] / ser_tot[i]
        global analysis_su, analysis_cases
        if fus_acc > 0 and fus_acc / sep_acc <10:
            analysis_su = analysis_su + (fus_acc/sep_acc)
            analysis_cases = analysis_cases+1
            print("===> ", sep_acc/fus_acc)
        if sep_total == 0:
            data_array[idx, clstr_no] = 0
            data_array[idx + 1, clstr_no] = 0
            data_array[idx + 2, clstr_no] = 0
            data_array[idx + 3, clstr_no] = 0
            data_array[idx + 4, clstr_no] = 0
            data_array[idx + 5, clstr_no] = 0
            data_array[idx + 6, clstr_no] = 0
            data_array[idx + 7, clstr_no] = 0
        else:
            #data_array[idx, clstr_no] = analysis1[i] / sep_acc
            #data_array[idx + 1, clstr_no] = sep_total / sep_acc
            ser_to_fuse = ser_tot[i] / fus_acc
            ser_to_sep = ser_tot[i] / sep_acc
            data_array[idx, clstr_no] = ser_to_sep * (analysis1[i] / sep_acc)
            data_array[idx + 1, clstr_no] = ser_to_sep * (sep_total / sep_acc)
            if fus_acc == 0:  # hack for spldl
                data_array[idx + 2, clstr_no] = 0
                data_array[idx + 3, clstr_no] = 0
                data_array[idx + 4, clstr_no] = 0
                data_array[idx + 5, clstr_no] = 0
                data_array[idx + 6, clstr_no] = 0
                data_array[idx + 7, clstr_no] = 0
            else:
                data_array[idx + 2, clstr_no] = ser_to_fuse*(analysis_fused[i]/fus_acc)
                data_array[idx + 3, clstr_no] = ser_to_fuse * (fuse_t[i] / fus_acc)
                #data_array[idx + 4, clstr_no] = sep_to_lib * (lib_analysis[i] / lib_acc)
                data_array[idx + 4, clstr_no] = ser_tot[i] / lib_acc
                data_array[idx + 5, clstr_no] = ser_tot[i] / ls_acc
                data_array[idx + 6, clstr_no] = ser_tot[i] / lbc_acc
                data_array[idx + 7, clstr_no] = ser_tot[i] / dagp_acc


def pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, lib_t, lib_analysis,
                               analysis1,
                           analysis_fused, data_array, clstr_no,
                           num_part_per_matrix, num_mat):
    for i in range(num_mat):
        idx = num_part_per_matrix * i
        sep_total = sep1_tp1[i] + sep1_tp2[i]
        sep_acc = sep_total + analysis1[i]
        fus_acc = analysis_fused[i] + fuse_t[i]
        lib_acc = lib_t[i] + lib_analysis[i]
        if sep_total == 0:
            data_array[idx, clstr_no] = 0
            data_array[idx + 1, clstr_no] = 0
            data_array[idx + 2, clstr_no] = 0
            data_array[idx + 3, clstr_no] = 0
            data_array[idx + 4, clstr_no] = 0
            data_array[idx + 5, clstr_no] = 0
        else:
            data_array[idx, clstr_no] = analysis1[i] / sep_acc
            data_array[idx + 1, clstr_no] = sep_total / sep_acc
            sep_to_fuse = sep_acc / fus_acc
            sep_to_lib = sep_acc / lib_acc
            if fus_acc == 0:
                data_array[idx + 2, clstr_no] = 0
                data_array[idx + 3, clstr_no] = 0
                data_array[idx + 4, clstr_no] = 0
                data_array[idx + 5, clstr_no] = 0
            else:
                data_array[idx + 2, clstr_no] = sep_to_fuse*(analysis_fused[i]/fus_acc)
                data_array[idx + 3, clstr_no] = sep_to_fuse * (fuse_t[i] / fus_acc)
                data_array[idx + 4, clstr_no] = sep_to_lib * (lib_analysis[i] / lib_acc)
                data_array[idx + 5, clstr_no] = sep_to_lib * (lib_t[i] / lib_acc)


def pack_to_array_analysis(sep1_tp1, sep1_tp2, fuse_t, analysis1,
                           analysis_fused, data_array, clstr_no,
                  num_part_per_matrix, num_mat):
    for i in range(num_mat):
        idx = num_part_per_matrix * i
        sep_total = sep1_tp1[i] + sep1_tp2[i]
        sep_acc = sep_total + analysis1[i]
        fus_acc = analysis_fused[i] + fuse_t[i]
        global analysis_su, analysis_cases
        if fus_acc > 0 and fus_acc / sep_acc <10:
            analysis_su = analysis_su + (fus_acc/sep_acc)
            analysis_cases = analysis_cases+1
            print("===> ", sep_acc/fus_acc)
        if sep_total == 0:
            data_array[idx, clstr_no] = 0
            data_array[idx + 1, clstr_no] = 0
            data_array[idx + 2, clstr_no] = 0
            data_array[idx + 3, clstr_no] = 0
        else:
            data_array[idx, clstr_no] = analysis1[i] / sep_acc
            data_array[idx + 1, clstr_no] = sep_total / sep_acc
            sep_to_fuse = sep_acc / fus_acc
            if fus_acc == 0:
                data_array[idx + 2, clstr_no] = 0
                data_array[idx + 3, clstr_no] = 0
            else:
                data_array[idx + 2, clstr_no] = sep_to_fuse*(analysis_fused[i]/fus_acc)
                data_array[idx + 3, clstr_no] = sep_to_fuse * (fuse_t[i] / fus_acc)


def get_mkl_time(f, header):
    f = f.split(".")[0] + "_mkl.csv"
    print(preprocess_text_file(f))
    csv_dic = pd.read_csv(f)
    csv_dic1 = csv_dic.sort_values("A Nonzero", ascending=False)
    mat_name = c_labs[Index.MAT_NAME.value]
    #csv_dic = csv_dic1.sort_values(mat_name, ascending=True)
    mkl_time = csv_dic1[header].values
    mkl_analysis = csv_dic1[header + ' Analysis Time'].values
    return mkl_time, mkl_analysis

def get_dagp_time(path, type, mat_list):
    csv_file = pd.read_csv(path)
    lab1 = c_labs[Index.MAT_NAME.value]  # key label
    sorted_dic = preprocess_csv(csv_file, lab1, lab1)
    out_list = []
    out_list_analysis = []
    for m in mat_list:
        out_list.append(sorted_dic[sorted_dic["Matrix Name"] == m][type].values[0])
        out_list_analysis.append(sorted_dic[sorted_dic["Matrix Name"] == m][type + " Analysis"].values[0])
    out_list = np.array(out_list)
    out_list_analysis = np.array(out_list_analysis)
    return out_list, out_list_analysis

def plot_stacked_bar_for_files(input_files, to_plot=True):
    num_cluster = 7
    num_matrix = 8
    num_part_per_matrix = 7  # for executor
    num_part_per_matrix_analysis = 8 #4
    data_array = np.zeros((num_matrix*num_part_per_matrix, num_cluster))
    global newdata_array
    newdata_array = np.zeros((num_matrix*(num_part_per_matrix-1), num_cluster))
    data_array_analysis = np.zeros((num_matrix * num_part_per_matrix_analysis,
                                    num_cluster))
    global num_times_amortize
    num_part_per_matrix_analysis_amortiz = 5
    num_times_amortize = np.zeros((num_matrix * num_part_per_matrix_analysis_amortiz,
                                    num_cluster))
    global ls_fv, ls_f_av, lbc_fv, lbc_f_av, dagp, dagp_a
    clstr_no = 0
    x_axis_second = []
    mat_name_order = []
    mkl_hdr_txt = "Parallel MKL Non-fused"
    orig_name = []
    dagp_path = ""
    input_files = sorted(input_files)
    for f in input_files:
        if "dagp_" in f:
            dagp_path = f
    for f in input_files:
        if "mkl" in f or "dagp" in f:
            continue
        print(preprocess_text_file(f))
        csv_dic1 = pd.read_csv(f)
        mat_name = c_labs[Index.MAT_NAME.value]
        csv_dic = csv_dic1.sort_values("A Nonzero", ascending=False)
        if len(csv_dic) == 0:
            return 0
        if clstr_no == 0:
            mat_name_order = csv_dic[mat_name].values
        else:
            if sum(mat_name_order != csv_dic[mat_name].values) != 0:
                print("ERRRRRRRRORRRRRRRRRRRRRRRRR")
        report_type = csv_dic[c_labs[Index.CODE_TYPE.value]].values[0].upper()
        mat_list = csv_dic["Matrix Name"].values
        print(mat_list)
        if report_type == "MV-TRSV":
            if "CC.csv" in f:
                [dagp, dagp_a] = get_dagp_time(dagp_path, report_type + " CC", mat_list)
            else:
                [dagp, dagp_a]= get_dagp_time(dagp_path, report_type + " RR", mat_list)
        else:
            [dagp, dagp_a]= get_dagp_time(dagp_path, report_type, mat_list)
        op_type = report_type_dic[report_type]
        print(report_type, sep=',')
        ser_time = csv_dic["Serial Non-fused"].values
        #if report_type == "MV-TRSV" or report_type == "ILU0-TRSV" or report_type == "DAD-IC0":
        #    continue
        if report_type == "MV-MV" or report_type == "MV-MV DIFF":
            # sep1 = "Parallel Non-fused CSC-CSC"
            # par1 = "Parallel Fused CSC-CSC"
            # lib = mkl_hdr_txt + " CSC-CSC"
            # mkl_t, mkl_a = get_mkl_time(f, lib)
            # sep1_tp1 = csv_dic[sep1 + ' p1'].values
            # sep1_tp2 = csv_dic[sep1 + ' p2'].values
            # fuse_t = csv_dic[par1].values
            # pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
            #               num_part_per_matrix, num_matrix)
            # sep1_analysis = np.zeros(num_matrix)  # no analysis time
            # fuse_analysis = csv_dic[par1 + ' Analysis Time'].values
            # pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
            #                        fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
            #                        num_part_per_matrix_analysis, num_matrix)
            # # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
            # #                        fuse_analysis, data_array_analysis, clstr_no,
            # #                        num_part_per_matrix_analysis, num_matrix)
            # x_axis_second.append(op_type + '\nCSC-CSC')
            # clstr_no += 1
            if "CC.csv" in f:
                sep1 = "Parallel Non-fused CSC-CSC"
                par1 = "Parallel Fused CSC-CSC"
                lib = mkl_hdr_txt + " CSC-CSC"
                mkl_t, mkl_a = get_mkl_time(f, lib)
                # csv_dic_gp = extract_tuned(csv_dic)
                sep1_tp1 = csv_dic[sep1 + ' p1'].values
                sep1_tp2 = csv_dic[sep1 + ' p2'].values
                fuse_t = csv_dic[par1].values
                pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                              num_part_per_matrix, num_matrix)
                sep1_analysis = np.zeros(num_matrix)  # no analysis time
                fuse_analysis = csv_dic[par1 + ' Analysis Time'].values
                pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
                # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
                #                        fuse_analysis, data_array_analysis, clstr_no,
                #                        num_part_per_matrix_analysis, num_matrix)
                x_axis_second.append(op_type + '\nCSC-CSC')
            else:
                sep1 = "Parallel Non-fused CSR-CSR"
                par1 = "Parallel Fused CSR-CSR"
                lib = mkl_hdr_txt + " CSR-CSR"
                mkl_t, mkl_a = get_mkl_time(f, lib)
                sep1_tp1 = csv_dic[sep1 + ' p1'].values
                sep1_tp2 = csv_dic[sep1 + ' p2'].values
                fuse_t = csv_dic[par1].values
                pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                              num_part_per_matrix, num_matrix)
                sep1_analysis = np.zeros(num_matrix)  # no analysis time
                fuse_analysis = csv_dic[par1 + ' Analysis Time'].values
                pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
                # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
                #                        fuse_analysis, data_array_analysis, clstr_no,
                #                        num_part_per_matrix_analysis, num_matrix)
                x_axis_second.append(op_type + '\nCSR-CSR')
            clstr_no += 1

        elif report_type == "MV-MV IND":
            sep1 = "Parallel Non-fused CSR-CSR"
            par1 = "Parallel Fused CSR-CSR"
            lib = mkl_hdr_txt + " CSR-CSR"
            mkl_t, mkl_a = get_mkl_time(f, lib)
            sep1_tp1 = csv_dic[sep1 + ' p1'].values
            sep1_tp2 = csv_dic[sep1 + ' p2'].values
            fuse_t = csv_dic[par1].values
            pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                          num_part_per_matrix, num_matrix)
            sep1_analysis = np.zeros(num_matrix)  # no analysis time
            fuse_analysis = csv_dic[par1 + ' Analysis Time'].values
            pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
            # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
            #                        fuse_analysis, data_array_analysis, clstr_no,
            #                        num_part_per_matrix_analysis, num_matrix)
            x_axis_second.append(op_type + '\nCSR-CSR')
            clstr_no += 1

        elif report_type == "TRSV-MV":
            sep1 = "Parallel LBC Non-fused CSR-CSC"
            par1 = "Parallel Fused CSR-CSC NOR"
            par1_tmp = "Parallel Fused CSR-CSC"
            lib = mkl_hdr_txt + " CSR-CSC IE"
            ls = "Parallel Fused CSR-CSC Joint-DAG LS"
            ls_a = "Parallel Fused CSR-CSC Analysis Time Joint-DAG LS"
            lbc_fus = "Parallel Fused CSR-CSC Joint-DAG LBC"
            lbc_fus_a = "Parallel Fused CSR-CSC Analysis Time Joint-DAG LBC"
            mkl_t, mkl_a = get_mkl_time(f, lib)
            sep1_tp1 = csv_dic[sep1 + ' p1'].values
            sep1_tp2 = csv_dic[sep1 + ' p2'].values
            #fuse_t = csv_dic[par1 + " NOR"].values
            fuse_t = csv_dic[par1].values
            #fuse_t_tmp = csv_dic[par1_tmp].values
            #fuse_t = np.minimum(fuse_t, fuse_t_tmp)
            ls_fv = csv_dic[ls].values
            ls_f_av = csv_dic[ls_a].values
            lbc_fv = csv_dic[lbc_fus].values
            lbc_f_av = csv_dic[lbc_fus_a].values
            pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                          num_part_per_matrix, num_matrix)
            sep1_analysis = csv_dic['Parallel LBC Non-Fused Analysis Time'].values
            fuse_analysis = csv_dic['Parallel Fused CSR-CSC Analysis Time'].values
            pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
            # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
            #                        fuse_analysis, data_array_analysis, clstr_no,
            #                        num_part_per_matrix_analysis, num_matrix)
            x_axis_second.append(op_type + '\nCSR-CSC')
            clstr_no += 1
        elif report_type == "TRSV-TRSV":
            sep1 = "Parallel LBC Non-fused CSR-CSR"
            par1 = "Parallel Fused CSR-CSR"
            lib = mkl_hdr_txt + " CSR-CSR IE"
            ls = "Parallel Fused CSR-CSR Joint-DAG Levelset"
            ls_a = "Parallel Fused CSR-CSR Analysis Time Joint-DAG Levelset"
            lbc_fus = "Parallel Fused CSR-CSR Joint-DAG LBC"
            lbc_fus_a = "Parallel Fused CSR-CSR Analysis Time Joint-DAG LBC"
            mkl_t, mkl_a = get_mkl_time(f, lib)
            sep1_tp1 = csv_dic[sep1 + ' p1'].values
            sep1_tp2 = csv_dic[sep1 + ' p2'].values
            fuse_t = csv_dic[par1].values
            ls_fv = csv_dic[ls].values
            ls_f_av = csv_dic[ls_a].values
            lbc_fv = csv_dic[lbc_fus].values
            lbc_f_av = csv_dic[lbc_fus_a].values
            pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                          num_part_per_matrix, num_matrix)
            sep1_analysis = csv_dic['Parallel LBC Non-fused CSR-CSR Analysis Time'].values
            fuse_analysis = csv_dic[par1 + ' Analysis Time'].values
            pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
            # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
            #                        fuse_analysis, data_array_analysis, clstr_no,
            #                        num_part_per_matrix_analysis, num_matrix)
            x_axis_second.append(op_type + '\nCSR-CSR')
            clstr_no += 1
        elif report_type == "MV-TRSV":
            if "CC.csv" in f:
                continue
                sep1 = "Parallel LBC Non-fused CSC-CSC"
                par1 = "Parallel Fused CSC-CSC"
                lib = mkl_hdr_txt + " CSC-CSC"
                ls = "Parallel Fused CSC-CSC Joint-DAG LS"
                ls_a = "Parallel Fused CSC-CSC Analysis Time Joint-DAG LS"
                lbc_fus = "Parallel Fused CSR-CSR Joint-DAG LBC"
                lbc_fus_a = "Parallel Fused CSR-CSR Analysis Time Joint-DAG LBC"
                mkl_t, mkl_a = get_mkl_time(f, lib)
                # csv_dic_gp = extract_tuned(csv_dic)
                sep1_tp1 = csv_dic[sep1 + ' p1'].values
                sep1_tp2 = csv_dic[sep1 + ' p2'].values
                fuse_t = csv_dic[par1].values
                ls_fv = csv_dic[ls].values
                ls_f_av = csv_dic[ls_a].values
                lbc_fv = csv_dic[lbc_fus].values
                lbc_f_av = csv_dic[lbc_fus_a].values
                pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                              num_part_per_matrix, num_matrix)
                sep1_analysis = csv_dic['Parallel LBC Non-fused Analysis Time'].values
                fuse_analysis = csv_dic[par1 + ' Analysis Time'].values
                pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
                # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
                #                        fuse_analysis, data_array_analysis, clstr_no,
                #                        num_part_per_matrix_analysis, num_matrix)
                x_axis_second.append(op_type + '\nCSC-CSC')
            else:
                sep1 = "Parallel LBC Non-fused CSR-CSR"
                par1 = "Parallel Fused CSR-CSR"
                lib = mkl_hdr_txt + " CSR-CSR IE"
                ls = "Parallel Fused CSR-CSR Joint-DAG LS"
                ls_a = "Parallel Fused CSR-CSR Analysis Time Joint-DAG LS"
                lbc_fus = "Parallel Fused CSR-CSR Joint-DAG LBC"
                lbc_fus_a = "Parallel Fused CSR-CSR Analysis Time Joint-DAG LBC"
                mkl_t, mkl_a = get_mkl_time(f, lib)
                sep1_tp1 = csv_dic[sep1 + ' p1'].values
                sep1_tp2 = csv_dic[sep1 + ' p2'].values
                fuse_t = csv_dic[par1].values
                ls_fv = csv_dic[ls].values
                ls_f_av = csv_dic[ls_a].values
                lbc_fv = csv_dic[lbc_fus].values
                lbc_f_av = csv_dic[lbc_fus_a].values
                pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                              num_part_per_matrix, num_matrix)
                sep1_analysis = csv_dic['Parallel LBC Non-fused Analysis Time'].values
                fuse_analysis = csv_dic[par1 + ' Analysis Time'].values
                pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
                # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
                #                        fuse_analysis, data_array_analysis, clstr_no,
                #                        num_part_per_matrix_analysis, num_matrix)
                x_axis_second.append(op_type + '\nCSR-CSR')
            clstr_no += 1

        elif report_type == "IC0-TRSV":
            sep1 = "Parallel LBC Non-fused CSC-CSC"
            par1 = "Parallel Fused Code CSC-CSC TW"
            ls = "Parallel Fused Code CSC-CSC Joint-DAG LS"
            ls_a = "Parallel Fused Code CSC-CSC Analysis Time Joint-DAG LS"
            lbc_fus = "Parallel Fused Code CSC-CSC Joint-DAG LBC"
            lbc_fus_a = "Parallel Fused Code CSC-CSC Analysis Time Joint-DAG LBC"
            lib = mkl_hdr_txt + " CSR-CSR"
            #mkl_t, mkl_a = get_mkl_time(f, lib)
            #mkl_t = np.zeros(num_matrix)
            mkl_t = ser_time
            mkl_a = np.zeros(num_matrix)
            sep1_tp1 = csv_dic[sep1 + ' p1'].values
            sep1_tp2 = csv_dic[sep1 + ' p2'].values
            fuse_t = csv_dic[par1].values
            ls_fv = csv_dic[ls].values
            ls_f_av = csv_dic[ls_a].values
            lbc_fv = csv_dic[lbc_fus].values
            lbc_f_av = csv_dic[lbc_fus_a].values
            pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                          num_part_per_matrix, num_matrix)
            sep1_analysis = csv_dic[sep1 + ' Analysis Time'].values
            fuse_analysis = csv_dic[sep1 + ' Analysis Time'].values
            pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
            # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
            #                        fuse_analysis, data_array_analysis, clstr_no,
            #                        num_part_per_matrix_analysis, num_matrix)
            x_axis_second.append(op_type + '\nCSR-CSR')
            clstr_no += 1
        elif report_type == "DAD-IC0":
            sep1 = "Parallel LBC Non-fused CSR-CSR"
            par1 = "Parallel Fused Code CSR-CSR"
            ls = "Parallel Fused Joint-DAG LS CSR-CSR"
            ls_a = "Parallel Fused Joint-DAG LS CSR-CSR Analysis Time"
            lbc_fus = "Parallel Fused Joint-DAG LBC CSR-CSR"
            lbc_fus_a = "Parallel Fused Joint-DAG LBC CSR-CSR Analysis Time"
            lib = mkl_hdr_txt + " CSR-CSR"
            #mkl_t, mkl_a = get_mkl_time(f, lib)
            #mkl_t = np.zeros(num_matrix)
            mkl_t = ser_time
            mkl_a = np.zeros(num_matrix)
            sep1_tp1 = csv_dic[sep1 + ' p1'].values
            sep1_tp2 = csv_dic[sep1 + ' p2'].values
            fuse_t = csv_dic[par1].values
            ls_fv = csv_dic[ls].values
            ls_f_av = csv_dic[ls_a].values
            lbc_fv = csv_dic[lbc_fus].values
            lbc_f_av = csv_dic[lbc_fus_a].values
            pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                          num_part_per_matrix, num_matrix)
            sep1_analysis = csv_dic[sep1 + ' Analysis Time'].values
            fuse_analysis = csv_dic["Parallel Fused CSR-CSR Analysis Time"].values
            pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
            # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
            #                        fuse_analysis, data_array_analysis, clstr_no,
            #                        num_part_per_matrix_analysis, num_matrix)
            x_axis_second.append(op_type + '\nCSC-CSC')
            clstr_no += 1
            # speed_up_graph(csv_dic, out_path+report_type+'csr-csr-speedup', sep1, par1, report_type+' CSR-CSR')
        elif report_type == "ILU0-TRSV":
            sep1 = "Parallel LBC Non-fused CSR-CSR"
            par1 = "Parallel Fused Code CSR-CSR"
            ls = "Parallel Fused Joint-DAG LS CSR-CSR"
            ls_a = "Parallel Fused Joint-DAG LS CSR-CSR Analysis Time"
            lbc_fus = "Parallel Fused Joint-DAG LBC CSR-CSR"
            lbc_fus_a = "Parallel Fused Joint-DAG LBC CSR-CSR Analysis Time"
            lib = mkl_hdr_txt + " CSR-CSR"
            mkl_t, mkl_a = get_mkl_time(f, lib)
            # csv_dic_gp = extract_tuned(csv_dic)
            # speed_up_graph(csv_dic_gp, out_path + report_type + 'csr-csr-speedup', sep1, par1, report_type + ' CSR-CSR')
            sep1_tp1 = csv_dic[sep1 + ' p1'].values
            sep1_tp2 = csv_dic[sep1 + ' p2'].values
            fuse_t = csv_dic[par1].values
            ls_fv = csv_dic[ls].values
            ls_f_av = csv_dic[ls_a].values
            lbc_fv = csv_dic[lbc_fus].values
            lbc_f_av = csv_dic[lbc_fus_a].values
            pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                          num_part_per_matrix, num_matrix)
            sep1_analysis = csv_dic[sep1 + ' Analysis Time'].values
            fuse_analysis = csv_dic[sep1 + ' Analysis Time'].values
            pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
            # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
            #                        fuse_analysis, data_array_analysis, clstr_no,
            #                        num_part_per_matrix_analysis, num_matrix)
            x_axis_second.append(op_type + '\nCSR-CSR')
            clstr_no += 1
        elif report_type == "DAD-ILU0":
            sep1 = "Parallel LBC Non-fused CSR-CSR"
            par1 = "Parallel Fused Code CSR-CSR"
            ls = "Parallel Fused Joint-DAG LS CSR-CSR"
            ls_a = "Parallel Fused Joint-DAG LS CSR-CSR Analysis Time"
            lbc_fus = "Parallel Fused Joint-DAG LBC CSR-CSR"
            lbc_fus_a = "Parallel Fused Joint-DAG LBC CSR-CSR Analysis Time"
            lib = mkl_hdr_txt + " CSR-CSR"
            mkl_t, mkl_a = get_mkl_time(f, lib)
            # csv_dic_gp = extract_tuned(csv_dic)
            # speed_up_graph(csv_dic_gp, out_path + report_type + 'csr-csr-speedup', sep1, par1, report_type + ' CSR-CSR')
            sep1_tp1 = csv_dic[sep1 + ' p1'].values
            sep1_tp2 = csv_dic[sep1 + ' p2'].values
            fuse_t = csv_dic[par1].values
            ls_fv = csv_dic[ls].values
            ls_f_av = csv_dic[ls_a].values
            lbc_fv = csv_dic[lbc_fus].values
            lbc_f_av = csv_dic[lbc_fus_a].values
            pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                          num_part_per_matrix, num_matrix)
            sep1_analysis = csv_dic[sep1 + ' Analysis Time'].values
            fuse_analysis = csv_dic["Parallel Fused CSR-CSR Analysis Time"].values
            pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
            # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
            #                        fuse_analysis, data_array_analysis, clstr_no,
            #                        num_part_per_matrix_analysis, num_matrix)
            x_axis_second.append(op_type + '\nCSR-CSR')
            clstr_no += 1
        elif report_type == "LDL-TRSV":
            sep1 = "Parallel LBC Non-fused CSC-CSC"
            par1 = "Parallel Fused Code TW CSC-CSC"
            lib = mkl_hdr_txt + " CSC-CSC"
            mkl_t, mkl_a = get_mkl_time(f, lib)
            sep1_tp1 = csv_dic[sep1 + ' p1'].values
            sep1_tp2 = csv_dic[sep1 + ' p2'].values
            fuse_t = csv_dic[par1].values
            pack_to_array_withlib_serial(ser_time, sep1_tp1, sep1_tp2, fuse_t, mkl_t, data_array, clstr_no,
                          num_part_per_matrix, num_matrix)
            sep1_analysis = csv_dic[sep1 + ' Analysis Time'].values
            # the same as one kernel
            fuse_analysis = csv_dic[sep1 + ' Analysis Time'].values
            pack_to_array_analysis_mkl_ser(ser_time, sep1_tp1, sep1_tp2, fuse_t, sep1_analysis,
                                           fuse_analysis, mkl_t, mkl_a, data_array_analysis, clstr_no,
                                           num_part_per_matrix_analysis, num_matrix)
           # pack_to_array_analysis_mkl(sep1_tp1, sep1_tp2, fuse_t, mkl_t, mkl_a, sep1_analysis,
           #                        fuse_analysis, data_array_analysis, clstr_no,
           #                        num_part_per_matrix_analysis, num_matrix)
            x_axis_second.append(op_type + '\nCSC-CSC')
            clstr_no += 1

        else:
            print("Type is not supported.\n")
            return 0
    if not to_plot:
        return data_array, mat_name
    print(" average exe speedup over mkl", mkl_to_fuse_avg/total_cases_lib)
    print(" average exe speedup over parsy", parsy_to_fuse_avg/total_cases)
    print(" average exe speedup over ls", ls_to_fuse_avg/total_cases)
    print(" average exe speedup over lbc", lbc_to_fuse_avg/total_cases)
    print(" average exe speedup over dagp", dagp_to_fuse_avg/total_cases)
    print(" max exe speedup over parsy", parsy_max_su)
    print(" min exe speedup over parsy", parsy_min_su)
    print(" max exe speedup over mk", mkl_max_su)
    print(" min exe speedup over mk", mkl_min_su)
    #print(" average exe speedup over Parsy inspector", analysis_su/analysis_cases)
    idx = [['ParSy K1', 'ParSy K2'], ['Sparse Fusion'], ['MKL'], ['Fused Wavefront'], ['Fused LBC'], ['Fused DAGp'] ]
    x_axis_label = ['1', '2', '3', '4', '5', '6', '7', '8'
                #    '9', '10', '11', '12' #, '5', '6', '7', '8',
    #                '1', '2', '3'
                    ]  # matrix labels
    colors1 = ('indigo', 'skyblue', 'red', 'teal', 'gray', 'saddlebrown', 'purple')

    # clustered_stacked_bar_graph(data_array, idx, num_cluster,
    #                             'Sequential Time / Implementation Time', '', x_axis_label,
    #                             x_axis_second, 'exec_perf_d', colors1)

    #t_idx = []
    #newdata_array = np.zeros((num_matrix*(num_part_per_matrix-1), num_cluster))
    #newdata_array[1:6, :] = data_array[2:7, :]
    #newdata_array[0, :] = sum(data_array[0:2, :])
    #newdata_array[0, :] = data_array[0, :] + data_array[1, :]
    idx = [['ParSy'], ['Sparse Fusion'], ['MKL'], ['Fused Wavefront'], ['Fused LBC'], ['Fused DAGp'] ]
    x_axis_label = ['1', '2', '3', '4', '5', '6', '7', '8'
                    #    '9', '10', '11', '12' #, '5', '6', '7', '8',
                    #                '1', '2', '3'
                    ]  # matrix labels
    colors1 = ('indigo', 'red', 'teal', 'gray', 'saddlebrown', 'purple')

    # clustered_stacked_bar_graph(newdata_array, idx, num_cluster,
    #                             'Sequential Time / Implementation Time', '', x_axis_label,
    #                             x_axis_second, 'exec_perf_d', colors1)


    coef_a = np.array([[1.52, 1.54, 0.45, 1.55, 0.61, 0.43, 0.61],
                       [1.5, 1.54, 0.45, 1.54, 0.61, 0.45, 0.61],
                       [1.4, 1.45, 0.47, 1.45, 0.48, 0.50, 0.47],
                       [1.47, 1.48, 0.72, 1.49, 0.50, 0.77, 0.47],
                       [1.42, 1.47, 0.45, 1.47, 0.51, 0.46, 0.49],
                       [0.91, 1.14, 0.17, 1.14, 0.33, 0.18, 0.32],
                       [1.47, 1.50, 0.73, 1.49, 0.49, 0.77, 0.48],
                       [1.41, 1.70, 0.89, 1.70, 0.44, 0.76, 0.42]])

    flops_bar = np.zeros((num_matrix * (num_part_per_matrix-1),
                          num_cluster))

    for k in range(num_cluster):
        for f in range(num_matrix):
            lb1 = f * (num_part_per_matrix-1)
            ub1 = (f+1) * (num_part_per_matrix-1)
            flops_bar[lb1:ub1, k] = newdata_array[lb1:ub1, k] * coef_a[f, k]

    flops_bar2 = np.zeros((num_matrix * (num_part_per_matrix - 2),
                          num_cluster))
    for k in range(num_cluster):
        for f in range(num_matrix):
            lb1 = f * (num_part_per_matrix - 1)
            ub1 = (f+1) * (num_part_per_matrix-1) - 2
            lb2 = f * (num_part_per_matrix - 2)
            ub2 = (f + 1) * (num_part_per_matrix-2) - 1
            flops_bar2[lb2:ub2, k] = flops_bar[lb1:ub1, k]
            flops_bar2[ub2, k] = min(flops_bar[ub1, k], flops_bar[ub1+1, k])
    idxf = [['ParSy'], ['Sparse Fusion'], ['MKL'], ['Fused Wavefront'], ['Best of Fused LBC-DAGP']]
    clustered_stacked_bar_graph(flops_bar2, idxf, num_cluster,
                                'GFLOP/s', '', x_axis_label,
                                x_axis_second, 'exec_perf_flops', colors1)

    idx_ana = [['ParSy inspector', 'ParSy executor'], ['FuSy inspector', 'FuSy executor'], ['MKL']]
    colors2 = ('olive', 'magenta', 'maroon', 'indigo','teal', 'khaki', 'purple')
    # clustered_stacked_bar_graph(data_array_analysis, idx_ana, num_cluster,
    #                             'Sequential / Tool Accumulated Time', '', x_axis_label,
    #                             x_axis_second, 'inspec_perf_d', colors2)
    idx_amort = [['ParSy'], ['Sparse Fusion'], ['Fused Wavefront'], ['Fused LBC'],  ['Fused DAGp']]
    colors3 = ('olive',  'magenta', 'maroon','darkgoldenrod', 'darkgreen')
    amort_new = np.zeros((num_matrix * num_part_per_matrix_analysis_amortiz,
                                    num_cluster))
#    for i in range(num_matrix*num_part_per_matrix_analysis_amortiz):
#        for j in range(num_cluster):
    amort_new = np.clip(num_times_amortize, -5, 80)
    # clustered_stacked_bar_graph(amort_new, idx_amort, num_cluster,
    #                             'Number of Executor Runs', '', x_axis_label,
    #                             x_axis_second, 'inspec_overhead', colors3)
    return data_array, mat_name


def main(argv):
    d1 = np.array([[0.5, 1.5],  # p1 of m1 type 1-2
                   [0.5, 0.25],  # p2 of m1 type 1-2
                   [1.2, 1.4],  # fused m1 type 1-2
                   [0.4, 0.6],  # p1 of m2 type 1-2
                   [0.6, 0.14],  # p2 of m2 type 1-2
                   [1.5, 2.6]]) # fused m2 type 1-2
    # this says the first two are stacked bar and we have two bars per cluster
    idx = [['ParSy P1', 'ParSy P2'], ['FuSy']]
    x_axis_label = ['1', '2']  #  matrix labels
    x_axis_second = ['ICHOL0-TRSV', 'MV-MV']
    clustered_stacked_bar_graph(d1, idx, 2, 'Speedup', '', x_axis_label,
                                x_axis_second)

def main2(argv):
    NAME = 'Name'
    matrix_list = ['af_0_k101', 'af_shell10', 'apache2']
#    in_csv = pd.read_csv("/home/kazem/Downloads/fusion/jointdag.csv")
    in_path = argv[0]
    errors = preprocess_text_file(in_path)
    in_csv_log = pd.read_csv(in_path)
    lab1 = c_labs[Index.MAT_NAME.value]  # key label
    in_csv = preprocess_csv(in_csv_log, lab1, lab1)
    mat_list = in_csv[lab1].values
    num_matrices = len(mat_list)
    num_stacks = 2
    dag_dag_l1 = 'Parallel Fused CSR-CSR Joint-DAG DAG Merging Time6'
    dag_dag_l2 = 'Parallel Fused CSR-CSR Joint-DAG DAG LBC Time6'
    elbc_l1 = 'Parallel Fused CSR-CSR JD E-LBC Analysis Time'
    elbc_l2 = 'Parallel Fused CSR-CSR JD E-LBC Metis Time'
    elbc_l3 = 'Parallel Fused CSR-CSR JD E-LBC Merging Time'
    d1 = np.zeros((11, num_matrices))
    idx = [["Joint-DAG Tree Merging", "Joint-DAG Tree Symmetrize", "Joint-DAG Tree Chordalize", "Joint-DAG Tree LBC"],
           ["Jonit DAG DAG Merging", "Joint DAG DAG LBC"],
           ["Enhanced LBC Merging", "Enhanced LBC Remaining Time", "Enhanced LBC METIS"],
           ["Iteration Fusion Remaining Time", "Iteration Fusion LBC"]]
    d1[0, :] = np.array(in_csv['Parallel Fused CSR-CSR Joint-DAG Tree Merging Time'].values)
    d1[1, :] = np.array(in_csv['Parallel Fused CSR-CSR Joint-DAG Tree Symmetrization Time'].values)
    d1[2, :] = np.array(in_csv['Parallel Fused CSR-CSR Joint-DAG Tree Chordalization Time'].values)
    dag_tree = np.array(in_csv['Parallel Fused CSR-CSR Joint-DAG Tree LBC Time'].values)
    d1[3, :] = dag_tree - (d1[1, :] + d1[2, :])  # np.array(in_csv['LBC Tree Time (sec)'].values)

    d1[4, :] = np.array(in_csv[dag_dag_l1].values)
    d1[5, :] = np.array(in_csv[dag_dag_l2].values)

    elbc_analysis = np.array(in_csv[elbc_l1].values)
    d1[6, :] = np.array(in_csv[elbc_l3].values)
    d1[8, :] = np.array(in_csv[elbc_l2].values)
    d1[7, :] = elbc_analysis - (d1[6, :] + d1[8, :])  # remaining time

    if_analysis = np.array(in_csv['Parallel Fused CSR-CSR Analysis Time'].values)
    d1[10, :] = np.array(in_csv['Parallel Fused CSR-CSR Analysis Time-LBC Tree'].values)
    d1[9, :] = if_analysis - d1[10, :]
    # tot = sum(d1[0:4,:])
    # print("Merg %", np.average(d1[0, :] / tot))
    # print("Sym %", np.average(d1[1, :] / tot))
    # print("Chor %", np.average(d1[2, :] / tot))
    # print("LBC %", np.average(d1[3, :] / tot))
    # tot2 = sum(d1[4:6, :])
    # print("Rem %", np.average(d1[4, :] / tot2))
    # print("LBC %", np.average(d1[5, :] / tot2))
    clustered_stacked_bar_graph(d1, idx, num_matrices, 'Time (seconds) ', '', mat_list, "")


def main3(argv):
    NAME = 'Name'
    matrix_list = ['af_0_k101', 'af_shell10', 'apache2']
    #    in_csv = pd.read_csv("/home/kazem/Downloads/fusion/jointdag.csv")
    in_path = argv[0]
    errors = preprocess_text_file(in_path)
    in_csv_log = pd.read_csv(in_path)
    lab1 = c_labs[Index.MAT_NAME.value]  # key label
    in_csv = preprocess_csv(in_csv_log, lab1, lab1)
    mat_list = in_csv[lab1].values
    num_matrices = len(mat_list)
    num_stacks = 2
    d1 = np.zeros((5, num_matrices))
    idx = [["Serial"],
           ["Joint-DAG Tree Executor"],
           ["Joint-DAG DAG Executor"],
           ["Enhanced LBC Executor"],
           ["Iteration Fusion Executor"]
          # ["Parallel Un-fused"]
           ]
    d1[0, :] = np.array(in_csv['Serial Non-fused'].values)
    d1[1, :] = np.array(in_csv['Parallel Fused CSR-CSR JDT'].values)
    d1[2, :] = np.array(in_csv['Parallel Fused CSR-CSR JDD6'].values)
    d1[3, :] = np.array(in_csv['Parallel Fused CSR-CSR JD E-LBC'].values)
    d1[4, :] = np.array(in_csv['Parallel Fused CSR-CSR IF1'].values)
    #d1[5, :] = np.array(in_csv['Parallel LBC Non-fused CSR-CSR p1'].values) + \
    #           np.array(in_csv['Parallel LBC Non-fused CSR-CSR p2'].values)

    clustered_stacked_bar_graph(d1, idx, num_matrices, 'Time (seconds) ', '', mat_list, "")


def main4(argv):
    NAME = 'Name'
    #    in_csv = pd.read_csv("/home/kazem/Downloads/fusion/jointdag.csv")
    in_path = argv[0]
    errors = preprocess_text_file(in_path)
    in_csv_log = pd.read_csv(in_path)
    lab1 = c_labs[Index.MAT_NAME.value]  # key label
    in_csv = preprocess_csv(in_csv_log, lab1, lab1)
    mat_list = in_csv[lab1].values
    num_matrices = len(mat_list)
    num_stacks = 2
    d1 = np.zeros((2, num_matrices))
    idx = [
           ["Enhanced LBC"],
           ["Iteration Fusion"]
           # ["Parallel Un-fused"]
           ]
    d1[0, :] = np.array(in_csv['Parallel Fused CSR-CSR JD E-LBC Redundant'].values)
    d1[1, :] = np.array(in_csv['Parallel Fused CSR-CSR IF Redundant'].values)
    #d1[5, :] = np.array(in_csv['Parallel LBC Non-fused CSR-CSR p1'].values) + \
    #           np.array(in_csv['Parallel LBC Non-fused CSR-CSR p2'].values)

    clustered_stacked_bar_graph(d1, idx, num_matrices, 'Redundant Iterations', '', mat_list, "")


def main5(argv):
    NAME = 'Name'
    #    in_csv = pd.read_csv("/home/kazem/Downloads/fusion/jointdag.csv")
    in_path = argv[0]
    errors = preprocess_text_file(in_path)
    in_csv_log = pd.read_csv(in_path)
    lab1 = c_labs[Index.MAT_NAME.value]  # key label
    in_csv = preprocess_csv(in_csv_log, lab1, lab1)
    mat_list = in_csv[lab1].values
    num_matrices = len(mat_list)
    num_stacks = 2
    d1 = np.zeros((4, num_matrices))
    idx = [
        ["Serial"],
        ["LBC Tree"],
        ["LBC DAG"],
        ["Enhanced LBC"]
    ]
    d1[0, :] = np.array(in_csv['Serial Non-fused'].values)
    d1[1, :] = np.array(in_csv['Parallel LBC Tree CSR'].values)
    d1[2, :] = np.array(in_csv['Parallel LBC DAG-6 CSR'].values)
    d1[3, :] = np.array(in_csv['Parallel LBC Enhanced-6 CSR'].values)
    #d1[5, :] = np.array(in_csv['Parallel LBC Non-fused CSR-CSR p1'].values) + \
    #           np.array(in_csv['Parallel LBC Non-fused CSR-CSR p2'].values)
    clustered_stacked_bar_graph(d1, idx, num_matrices, 'Time (sec)', '', mat_list, "")


def main6(argv):
    in_path = "/home/kazem/Dropbox/tmp/fusion/Draft01/graphing02_comet/log_final_parallel/MV-MV_CC.csv"
    lab1 = "Parallel Fused CSC-CSC"
    lab2 = "Parallel Fused CSR-CSR"
    lab_cmet = " METISCORS"
    lab_consec = " CONSEC"
    lab_bfs = " BFS"
    base_lab = "Serial Non-fused"
    in_csv_log1 = pd.read_csv(in_path)
    #in_csv_log3 = pd.read_csv(in_path)
    num_matrix = 8
    part_per_mat = 4
    num_cluster = 4
    data_array = np.zeros((num_matrix*part_per_mat, num_cluster))
    data_array_analysis = np.zeros((num_matrix*part_per_mat, num_cluster))
    cluster_no=0
    # MV-MV CSR-CSR
    data_array, data_array_analysis = get_data(cluster_no, data_array, data_array_analysis, in_csv_log1, lab2, lab_bfs, lab_cmet, lab_consec, num_matrix,
                          part_per_mat, base_lab, "Parallel Non-fused CSR-CSR")
    # MV-MV CSC CSC
    cluster_no += 1
    data_array, data_array_analysis = get_data(cluster_no, data_array, data_array_analysis, in_csv_log1, lab1, lab_bfs, lab_cmet, lab_consec, num_matrix,
                          part_per_mat, base_lab, "Parallel Non-fused CSC-CSC")
    # MV-MV Diff CSR CSR
    lab_cmet = " CMETIS"
    in_path = "/home/kazem/Dropbox/tmp/fusion/Draft01/graphing02_comet/log_final_parallel/MV-MV DIFF_CC.csv"
    in_csv_log1 = pd.read_csv(in_path)
    cluster_no += 1
    data_array, data_array_analysis = get_data(cluster_no, data_array, data_array_analysis, in_csv_log1, lab2, lab_bfs, lab_cmet, lab_consec, num_matrix,
                          part_per_mat, base_lab, "Parallel Non-fused CSR-CSR")

    cluster_no += 1
    data_array, data_array_analysis = get_data(cluster_no, data_array, data_array_analysis, in_csv_log1, lab1, lab_bfs, lab_cmet, lab_consec, num_matrix,
                          part_per_mat, base_lab, "Parallel Non-fused CSC-CSC")



    #num_cluster = 2
    idx = [['Unfused'], ['Fused, METIS-based Seed Partitioning'],
           #['Fused, Coarsened METIS Seed Partitioning'],
           ['Fused, Consecutive iteration partitioning'], ['Fused, BFS-based Seed partitioning']]
    x_axis_label = ['1', '2', '3', '4', '5', '6', '7', '8']  # matrix labels
    colors1 = ('skyblue', 'red', 'indigo', 'teal', 'black')
    x_axis_second = [r'$A^{2}*x$' '\nCSR-CSR', r'$A^{2}*x$' '\nCSC-CSC', r'$A*B*x$' '\nCSR-CSR', r'$A*B*x$' '\nCSC-CSC']
    clustered_stacked_bar_graph(data_array, idx, num_cluster,
                                'Sequential / Tool Time', '', x_axis_label,
                                x_axis_second, 'exec_perf_p', colors1)

    clustered_stacked_bar_graph(data_array_analysis, idx, num_cluster,
                                'Number of Runs', '', x_axis_label,
                                x_axis_second, 'ins_perf_p', colors1)


def get_data(cluster_no, data_array, data_array_analysis, in_csv_log1, lab1, lab_bfs, lab_cmet, lab_consec, num_matrix, part_per_mat,
             base_lab, unfused_lab):
    for i in range(num_matrix):
        j = i * part_per_mat
        basline = in_csv_log1[base_lab].values[i]
        data_array[j, cluster_no] = basline / (in_csv_log1[unfused_lab +" p1"].values[i] + in_csv_log1[unfused_lab + " p2"].values[i])
        data_array[j + 1, cluster_no] = basline / in_csv_log1[lab1].values[i]
        #data_array[j + 2, cluster_no] = basline / in_csv_log1[lab1 + lab_cmet].values[i]
        data_array[j + 2, cluster_no] = basline / in_csv_log1[lab1 + lab_consec].values[i]
        data_array[j + 3, cluster_no] = basline / in_csv_log1[lab1 + lab_bfs].values[i]

        at = " Analysis Time"
        a1 = in_csv_log1[lab1 + at].values[i]
        data_array_analysis[j + 1, cluster_no] = max(int(a1 / (basline - in_csv_log1[lab1].values[i])), 1)
        #a1 = in_csv_log1[lab1 + at + lab_cmet].values[i]
        #data_array_analysis[j + 2, cluster_no] = max(int(a1 / (basline - in_csv_log1[lab1 + lab_cmet].values[i])), 1)
        a1 = in_csv_log1[lab1 + at + lab_consec].values[i]
        data_array_analysis[j + 2, cluster_no] = max(int(a1 / (basline - in_csv_log1[lab1 + lab_consec].values[i])), 1)
        a1 = in_csv_log1[lab1 + at + lab_bfs].values[i]
        data_array_analysis[j + 3, cluster_no] = max(int(a1 / (basline - in_csv_log1[lab1 + lab_bfs].values[i])), 1)

    return data_array, data_array_analysis


if __name__ == "__main__":
    main6(sys.argv[1:])