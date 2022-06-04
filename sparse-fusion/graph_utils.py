__author__ = 'Kazem'
import os
from os.path import isfile, join, exists
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import gmean
from enum import Enum
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches

from numpy.polynomial.polynomial import polyfit

global_dict = {
    "ERRORS": 0,
    "TYPE": "",
    "A-SpeedUp": 0,
    "G-SpeedUp": 0,
    "Max-SpeedUp": 0,
    "Min-SpeedUp": 0,
    "IdealG-SpeedUp": 0,
    "Faster %": 0
}

# Preparing the graph info
# Other colors: https://matplotlib.org/gallery/color/named_colors.html
colors = ('red', 'skyblue', 'indigo', 'olive', 'slateblue', 'magenta',
          'slategray', 'limegreen', 'maroon', 'teal', 'khaki', 'purple',
          'r', 'c', 'm', 'y', 'k')
markers = ('s', 'o', 'v', ".", ",", "+", "^", "<", ">",
           "1", "2", "3", "4", "8", "s", "p", "P")

weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']

font0 = FontProperties()

#https://www.7-cpu.com/cpu/Haswell.html
comet_params = {'L1_ACCESS_TIME': 4, 'L2_ACCESS_TIME': 12, 'L3_ACCESS_TIME': 36, #or 43,
                'MAIN_MEMORY_ACCESS_TIME': 89}


def plot_bar_chart(all_data, legend_labels, x_axis_labels, y_axis_label,
                   out_file_name='', y_scale="linear"):
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

    width = 0.4
    max_y = np.amax(all_data)+0.5
    ind = (n_groups + 1) * width * np.arange(n_elements)
    x_label = np.arange(1, n_elements + 1)
    y_label = np.arange(0.5, max_y, 0.5)
    font_axis = font0.copy()
    metric_label = 'solve time(sec)'
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0, max_y])
    ax1.set_yscale(y_scale)
    for i in range(n_groups):
        ax1.bar(ind + i * width, all_data[i, :], width,
                label=legend_labels[i], color=colors[i],
                align='edge')

    font_axis.set_weight(weights[6])  # ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
    ax1.set_ylabel(y_axis_label, fontsize=14, fontproperties=font_axis)
    ax1.set_yticks(y_label, minor=False)
    ax1.axhline(y=[1], linestyle='--', color='gray')  #
    ax1.set_xticks(ind)
    ax1.set_xticklabels(x_axis_labels, fontsize=14)
    plt.xlim(-width)
    plt.xticks(rotation=55)
    plt.legend(loc='upper center', fontsize=20, ncol=2,  frameon=False)
    #plt.show()
    dir_out = os.path.dirname(out_file_name)
    base_out = os.path.basename(out_file_name).split('.')[0]
    if os.path.exists(dir_out):
        fig.savefig(join(dir_out, base_out + ".pdf"), bbox_inches='tight')
    else:
        print("Output path is not valid, saving current dir!")
        fig.savefig(base_out + ".pdf", bbox_inches='tight')


def plot_scatter_chart(all_data, legend_labels, x_axis_labels, y_axis_label,
                   out_file_name='', y_scale="linear"):
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

    width = 0.4
    max_y = np.amax(all_data)+0.5
    ind = (n_groups + 1) * width * np.arange(n_elements)
    x_label = np.arange(1, n_elements + 1)
    y_label = np.arange(0.5, max_y, 0.5)
    font_axis = font0.copy()
    metric_label = 'solve time(sec)'
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0, max_y])
    ax1.set_yscale(y_scale)
    for i in range(n_groups):
        ax1.scatter(x_label, all_data[i, :], s=10, color=colors[i],
                    marker=markers[i], label=legend_labels[i])

    font_axis.set_weight(weights[6])  # ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
    ax1.set_ylabel(y_axis_label, fontsize=14, fontproperties=font_axis)
    ax1.set_yticks(y_label, minor=False)
    ax1.axhline(y=[1], linestyle='--', color='gray')  #
    ax1.set_xticks(ind)
    ax1.set_xticklabels(x_axis_labels, fontsize=14)
    plt.xlim(-width)
    plt.xticks(rotation=55)
    plt.legend(loc='upper center', fontsize=20, ncol=2,  frameon=False)
    plt.show()
    dir_out = os.path.dirname(out_file_name)
    base_out = os.path.basename(out_file_name).split('.')[0]
    if os.path.exists(dir_out):
        fig.savefig(join(dir_out, base_out + ".pdf"), bbox_inches='tight')
    else:
        print("Output path is not valid, saving current dir!")
        fig.savefig(base_out + ".pdf", bbox_inches='tight')


### plots per mesh for all methods
def plot_line_graph(data_array, data_labels, axis_labels, title = '', out_file_name='', colors_in=np.array([]),
                    data_Ref=np.array([]), ref_label = '', y_scale = 'log'):
    if len(data_array) <= 0:
        print("all_data invalid.")
        return -1

    n_groups = len(data_array) #methods
    if len(data_labels) != n_groups:
        print("legend_labels or x_axis_labels are invalid.")
        return -1
    cm1 = plt.get_cmap('Set3')  ## some other values 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
    cm2 = plt.get_cmap('Dark2')
    colors1 = [cm1(1.*i/(n_groups+1)) for i in range(0, n_groups+1, 2)]  # NOTE: for determinitic color codes, this should be commented
    colors2 = [cm2(1.*i/(n_groups+1)) for i in range(0, n_groups+1, 2)]
    colors2.reverse()
    colors = [val for pair in zip(colors1, colors2) for val in pair]
    if len(colors_in) > 0:
        colors = colors_in
    width = 0.5
    constt = 0.5
    #y_label = np.arange(0, max_y, .2)
    font_axis = font0.copy()
    fig = plt.figure(figsize=(80, 30))
    ax1 = fig.add_subplot(111)
    ax1.set_title(title, fontsize=70)
    #ax1.text(100, 7500, r'Common Memory Access Percentage = 2%', fontsize=50, fontweight='bold')
    #max_y = np.amax(data_array)+0.5
    ax1.set_ylim([1, 1e7])
    ax1.set_xlim([1, 1600])
    #y_scale = 'linear'
    ax1.set_yscale(y_scale)
    for i in range(n_groups):
        tmp = np.array(data_array[i])
        if i == 1:
            ax1.plot(tmp, markers[i], linewidth=20, markersize=20, label=data_labels[i], color=colors[i], markeredgecolor='darkgoldenrod')
        else:
            ax1.plot(tmp, markers[i], linewidth=20, markersize=20, label=data_labels[i], color=colors[i], markeredgecolor='black')
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.tick_params(axis='y', which='both', left='off', right='off', color='w')
            ax1.tick_params(axis='x', which='both', left='off', right='off')
    #ax1.plot(data_array, markers[0], label=data_labels[0], color=colors[0])
    if len(data_Ref) > 0:
        ax1.plot(data_Ref, '-', linewidth=8, markersize=20, label=ref_label, color=colors[n_groups])
    #ax1.set_xlabel(axis_labels[0], fontsize=78, fontproperties=font_axis)
    #ax1.set_ylabel(axis_labels[1], fontsize=48, fontproperties=font_axis)
    #ax1.set_yticks(y_label, minor=False)
    #ax1.axhline(y=[1], linestyle='--', color='gray')  #
    #ax1.set_xticklabels(x_axis_labels, fontsize=28)plt.xlabel(axis_labels[0])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(axis='y', which='both', left='off', right='off')
    ax1.tick_params(axis='x', which='both', left='off', right='off')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(100)

    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(100)
    # x_tick_loc_second = [0.5, 1]
    # second_x_labels = ["AA", "BB"]
    # ax1.set_xticks(x_tick_loc_second, minor=False)
    # ax1.set_xticklabels(second_x_labels, fontsize=25)
    ax1.tick_params(axis='y', which='both', left='off', right='off')
    ax1.tick_params(axis='x', which='both', left='off', right='off')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    #ax1.spines['right'].set_color('white')
    plt.xlabel(axis_labels[0], fontsize=100, weight="bold")
    plt.ylabel(axis_labels[1], fontsize=100, weight="bold")
    plt.legend(loc='best', fontsize=110, ncol=1, frameon=False, prop={'size': 120})
    plt.show()
    dir_out = os.path.dirname(out_file_name)
    base_out = os.path.basename(out_file_name).split('.')[0]
    if os.path.exists(dir_out):
        fig.savefig(join(dir_out, base_out + ".pdf"), bbox_inches='tight')
    else:
        print("Output path is not valid, saving current dir!")
        fig.savefig(base_out + ".pdf", bbox_inches=0)
    plt.close()


def plot_line_graph_two_y(data_array, data_array_y2, data_labels, axis_labels, title = '',
                          out_file_name='', data_Ref=np.array([]), ref_label = '', y_scale = 'log'):
    if len(data_array) <= 0:
        print("all_data invalid.")
        return -1

    n_groups = len(data_array) #methods
    n_groups_y2 = len(data_array_y2)
    if len(data_labels) != n_groups + n_groups_y2:
        print("legend_labels or x_axis_labels are invalid.")
        return -1
    cm1 = plt.get_cmap('Set3')  ## some other values 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
    cm2 = plt.get_cmap('Dark2')
    n_groups_t = n_groups_y2 + n_groups
    colors1 = [cm1(1.*i/(n_groups_t+1)) for i in range(0, n_groups_t+1, 2)]  # NOTE: for determinitic color codes, this should be commented
    colors2 = [cm2(1.*i/(n_groups_t+1)) for i in range(0, n_groups_t+1, 2)]
    colors2.reverse()
    colors = [val for pair in zip(colors1, colors2) for val in pair]
    width = 0.5
    constt = 0.5
    #y_label = np.arange(0, max_y, .2)
    font_axis = font0.copy()
    fig = plt.figure(figsize=(50, 25))
    ax1 = fig.add_subplot(111)
    ax1.set_title(title, fontsize=50)
    #ax1.set_ylim([0, max_y])
    #y_scale = 'log'
    ax1.set_yscale(y_scale)
    for i in range(n_groups):
        tmp = np.array(data_array[i])
        ax1.plot(tmp, '-', linewidth=4, markersize=4, label=data_labels[i], color=colors[i])
    #ax1.plot(data_array, markers[0], label=data_labels[0], color=colors[0])
    if len(data_Ref) > 0:
        ax1.plot(data_Ref, '-', linewidth=8, markersize=8, label=ref_label, color=colors[n_groups])
    ax1.set_xlabel(axis_labels[0], fontsize=50)
    ax1.set_ylabel(axis_labels[1], fontsize=50)
    #ax1.set_yticks(y_label, minor=False)
    #ax1.axhline(y=[1], linestyle='--', color='gray')  #
    #ax1.set_xticklabels(x_axis_labels, fontsize=28)plt.xlabel(axis_labels[0])
    ax2 = ax1.twinx()
    for i in range(n_groups_y2):
        tmp = np.array(data_array_y2[i])
        ax2.plot(tmp, '-', linewidth=4, markersize=4, label=data_labels[n_groups+i], color=colors[n_groups+i])
    ax2.set_ylim(0, 100)
    #ax2.set_xlabel(axis_labels[0], fontsize=50, fontproperties=font_axis)
    ax2.set_ylabel(axis_labels[2], fontsize=50)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(50)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(50)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(50)

    #plt.xlabel(axis_labels[0], fontsize=50)
    #plt.ylabel(axis_labels[1], fontsize=50)
    #plt.legend(loc='best', fontsize=50, ncol=2, frameon=False, prop={'size': 60})
    ax1.legend(loc='upper left', fontsize=50, ncol=2, frameon=False, prop={'size': 60})
    ax2.legend(loc='upper right', fontsize=50, ncol=2, frameon=False, prop={'size': 60})
    plt.show()
    dir_out = os.path.dirname(out_file_name)
    base_out = os.path.basename(out_file_name).split('.')[0]
    if os.path.exists(dir_out):
        fig.savefig(join(dir_out, base_out + ".png"), bbox_inches='tight')
    else:
        print("Output path is not valid, saving current dir!")
        fig.savefig(base_out + ".png", bbox_inches='tight')
    plt.close()



def dic_to_csv(dict_data, csv_file):
    csv_columns = []
    header_exist = exists(csv_file)
    if not header_exist:
        try:
            with open(csv_file, 'w') as f:
                for key in sorted(dict_data.keys()):
                    csv_columns.append(key)
                    f.write("%s,"%(key))
        except IOError:
            print("I/O error")
    try:
        with open(csv_file, 'a') as f:
            f.write('\n')
            for key in sorted(dict_data.keys()):
                f.write("%s," % (dict_data[key]))
    except IOError:
        print("I/O error")


def exclude_items_by_key(log_lst, value_list, key):
    for v in value_list:
        index_names = log_lst[log_lst[key] == v].index
        # Delete these row indexes from dataFrame
        log_lst.drop(index_names, inplace=True)
    return log_lst

MATRIX_NAME='Matrix Name'
# column headers of the CSV file, Specific routines to fusion or ...?
c_labs = ['Code Type', 'Matrix Name', 'Ideal Speedup', 'LBC Param1', 'LBC Param2', 'Data Type',
          'NNZ Cost', 'Unit Cost', 'A Dimension']


class Index(Enum):
    CODE_TYPE = 0
    MAT_NAME = 1
    IDEAL_SU = 2
    LBC_P1 = 3
    LBC_P2 = 4
    VARIANT = 5
    NNZ_COST = 6
    UNIT_COST = 7
    DIMENSION = 8

#for initial results
#deleting_matrices = ['G3_circuit', 'ecology2', 'thermomech_dM', 'parabolic_fem', 'tmt_sym', 'apache2', 'thermal2']
# for cost model
#deleting_matrices = ['nd12k', 'nd24k','s3dkq4m2', 'cant', 'nasasrb', 'denormal', 'Pres_Poisson', 'bundle1', 'qa8fm',
#                     'msc10848', 'pwtk', 'thread', 'm_t1', 'cfd2', 'consph', 'boneS10', 'oilpan', 'vanbody', 'shipsec5',
#                     'x104', 'crankseg_2', 'thermomech_dM', 'hood', 'BenElechi1', 'bone010', 'bundle_adj']

deleting_matrices = [
    'thermomech_dM',
    'parabolic_fem', 'tmt_sym',
    'G3_circuit', 'apache2', 'crankseg_2',
    'thermal2',
    'pdb1HYS', 'offshore', 'cant', 'Dubcova3', 'cfd2', 'nasasrb',
    'ct20stif', 'vanbody', 'oilpan', 'qa8fm', '2cubes_sphere',
    'raefsky4', 'msc10848', 'denormal', 'bcsstk36', 'gyro',
    'msc23052', 'aft01', 'obstclae', 'nasa2910',
    'minsurfo', 'nd24k',
    'pwtk', 'shipsec5', 'smt', 's3rmt3m3', 'bcsstk16', 'bcsstk24',
    'ecology2',
     ##                good ones
   'nd12k',
                     'Muu',
     'Trefethen_20000',
     's3dkq4m2',
    'Kuu',
    'Pres_Poisson',
    'cbuckle',
    #chol0
    'olafu',
    'consph',
       'audikw_1',
    'm_t1', 'x104',
        'BenElechi1',
   'hood',
    #### second
    'bundle1',
    'fv2',
    'thread',
       'bmwcra_1',

       'ldoor',
        'boneS10',
   'msdoor',
   'af_shell7',

    #'Hook_1498',
    ###### first pick
    #   'StocF-1465',
   #'af_shell10',
   # 'ted_B_unscaled',
    #'bone010',
   # 'Flan_1565',
   #'af_0_k101',
    # 'Emilia_923',
   'Fault_639',
    'PFlow_742',
    'bundle_adj',
                     ]

#deleting_matrices = []

def strip_problem_name(p_name):
    name_only = os.path.basename(p_name).split('.')[0]
    return name_only


def preprocess_csv(csv_dic, problem_name, sorting_column_label):
    # make the problem names consistent
    n_elements = len(csv_dic[problem_name])
    for l in range(n_elements):
        csv_dic[problem_name][l] = strip_problem_name(csv_dic[problem_name][l])
    # deleting some entries
    csv_dic = exclude_items_by_key(csv_dic, deleting_matrices, c_labs[Index.MAT_NAME.value])
    # sorting
    #csv_dic_sorted = csv_dic.sort_values(by=[sorting_column_label])
    return csv_dic


def preprocess_text_file(in_path, del_str='!='):
    need_rewrite = False
    num_wrong = 0
    with open(in_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if del_str in line or line == '\n': # error or empty lines
            need_rewrite = True
            break
    if need_rewrite:
        with open(in_path, "w") as f:
            for line in lines:
                if not((del_str in line) or (line == '\n')):
                    f.write(line)
                else:
                    num_wrong += 1
    return num_wrong


def compute_speedups(sep1, sep2, fused):
    n_elem = len(sep1)
    t = np.zeros((2, n_elem))
    max_ideal = np.zeros((n_elem))
    for i in range(n_elem):
        max_ideal[i] = max(sep1[i], sep2[i])
        t[0, i] = (sep1[i] + sep2[i]) / max(sep1[i], sep2[i])
        t[1, i] = (sep1[i] + sep2[i]) / fused[i]
    mean_su = np.mean(t[1, :])
    gmean_su = gmean(t[1, :])
    mean_peak = gmean((max_ideal / fused))
    faster_percent = len(np.where(t[1, :] > 1)[0])/len(t[1,:])
    return t, mean_su, gmean_su, mean_peak, faster_percent


def compute_memory_cycle_for_one_group(row, arch_params):
    dl1_miss = row['PAPI_L1_DCM'].values
    dl2_miss = row['PAPI_L2_DCM'].values
    dl3_miss = row['PAPI_L3_TCM'].values
    dl1_access = row['PAPI_LST_INS'].values
    l1_mr = dl1_miss / dl1_access
    l2_mr = dl2_miss / dl1_miss
    l3_mr = dl3_miss / dl2_miss
    l1_access_cost = arch_params['L1_ACCESS_TIME']
    l2_access_cost = arch_params['L2_ACCESS_TIME']
    l3_access_cost = arch_params['L3_ACCESS_TIME']
    mm_access_cost = arch_params['MAIN_MEMORY_ACCESS_TIME']
    avg_mem_cycle = l1_access_cost + l1_mr*(l2_access_cost + l2_mr*(l3_access_cost + l3_mr*mm_access_cost))
    #avg_mem_cycle = (1-l1_mr) * l1_access_cost + (1-l2_mr) * l2_access_cost + (1-l3_mr) * l3_access_cost + l3_mr * mm_access_cost
    exec_cycle = avg_mem_cycle  * dl1_access
    return exec_cycle


def compute_locality_ratio_by_variant_matrix(csv_dic, variant1, variant2):
    mem_cycle_ratio = []
    cost_ratio = []
    mat_name = c_labs[Index.MAT_NAME.value]
    variant = c_labs[Index.VARIANT.value]
    cost_str = c_labs[Index.NNZ_COST.value]
    cost_str = c_labs[Index.UNIT_COST.value]
    gb = csv_dic.groupby([mat_name, variant])
    mat_list = csv_dic[mat_name].values
    i = 0
    while i < len(mat_list):
        v1_row = gb.get_group((mat_list[i], variant1))
        t1 = compute_memory_cycle_for_one_group(v1_row, comet_params)
        c1 = v1_row[cost_str].values
        v2_row = gb.get_group((mat_list[i], variant2))
        t2 = compute_memory_cycle_for_one_group(v2_row, comet_params)
        c2 = v2_row[cost_str].values
        mid = 2
        mem_cycle_ratio.append(t2[mid] / t1[mid])
        cost_ratio.append(1.0 / (c2[mid] / c1[mid]))
        print(mat_list[i], t2[mid] / t1[mid], (c2[mid] / c1[mid]), sep=',')
        # for j in range(v1_row.shape[0]):
        #     mem_cycle_ratio.append(t2[j]/t1[j])
        #     cost_ratio.append( (c2[j]/c1[j]))
        #     #mem_cycle_ratio.append(t2[j])
        #     #mem_cycle_ratio.append(t1[j])
        #     #cost_ratio.append(c2[j])
        #     #cost_ratio.append(c1[j])
        #     print(mat_list[i], t2[j]/t1[j], 1.0/(c2[j]/c1[j]), sep=',')
        assert v1_row.shape[0] == v2_row.shape[0]
        i += v1_row.shape[0]
    return np.array(mem_cycle_ratio), np.array(cost_ratio)


def extract_tuned(csv_dic):
    target_config = (4, 4000)  # use for selecting tuned one
    lbc_p1 = c_labs[Index.LBC_P1.value]
    lbc_p2 = c_labs[Index.LBC_P2.value]
    gb = csv_dic.groupby([lbc_p1, lbc_p2])
    csv_df_gb = gb.get_group(target_config)
    return csv_df_gb


def extract_tuned_tconfig(csv_dic, target_config):
    lbc_p1 = c_labs[Index.LBC_P1.value]
    lbc_p2 = c_labs[Index.LBC_P2.value]
    gb = csv_dic.groupby([lbc_p1, lbc_p2])
    csv_df_gb = gb.get_group(target_config)
    return csv_df_gb


def speed_up_graph(csv_df_gb, out_path, sep1, par1, legend_labels):
    sep1_tp1 = csv_df_gb[sep1 + ' p1'].values
    sep1_tp2 = csv_df_gb[sep1 + ' p2'].values
    fuse_t = csv_df_gb[par1].values
    [t, mean_su, gmean_su, mean_peak, faster] = compute_speedups(sep1_tp1, sep1_tp2, fuse_t)
    plot_bar_chart(t, [legend_labels+' Ideal Fused', legend_labels+' Fused'],
                   csv_df_gb[c_labs[Index.MAT_NAME.value]].values,
                   'Separate Time / Fused Time', out_path)
    global_dict["A-SpeedUp"] = mean_su
    global_dict["G-SpeedUp"] = gmean_su
    global_dict["Max-SpeedUp"] = np.max(t[1, :])
    global_dict["Min-SpeedUp"] = np.min(t[1, :])
    global_dict["Ideal-SpeedUp"] = mean_peak
    global_dict["Faster %"] = faster
    #print(mean_su, gmean_su, mean_peak, np.max(t[1, :]), sep=',')
    return csv_df_gb


def speed_up_graph_twogb(csv_df_gb, csv_df_gb_par1, out_path, sep1, par1, legend_labels):
    sep1_tp1 = csv_df_gb[sep1 + ' p1'].values
    sep1_tp2 = csv_df_gb[sep1 + ' p2'].values
    fuse_t = csv_df_gb_par1[par1].values
    [t, mean_su, gmean_su, mean_peak, faster] = compute_speedups(sep1_tp1, sep1_tp2, fuse_t)
    plot_bar_chart(t, [legend_labels+' Ideal Fused', legend_labels+' Fused'],
                   csv_df_gb[c_labs[Index.MAT_NAME.value]].values,
                   'Separate Time / Fused Time', out_path)
    #print(mean_su, gmean_su, mean_peak, np.max(t[1, :]), sep=',')
    global_dict["A-SpeedUp"] = mean_su
    global_dict["G-SpeedUp"] = gmean_su
    global_dict["Max-SpeedUp"] = np.max(t[1, :])
    global_dict["Min-SpeedUp"] = np.min(t[1, :])
    global_dict["Ideal-SpeedUp"] = mean_peak
    global_dict["Faster %"] = faster
    merged_df = csv_df_gb_par1.copy()
    merged_df[sep1 + ' p1'] = csv_df_gb[sep1 + ' p1'].values
    merged_df[sep1 + ' p2'] = csv_df_gb[sep1 + ' p2'].values
    return merged_df


def select_plotter(csv_dic, out_path, lib_file=False):
    merged_df = []
    export = True
    if len(csv_dic) == 0:
        return 0
    report_type = csv_dic[c_labs[Index.CODE_TYPE.value]].values[0].upper()
    export_path = join(out_path, "log_final_parallel")
    if "TRSV" in report_type or "IC0" in report_type or "ILU0" in report_type:
        export_path = join(out_path, "log_final_dependent")
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    lbc_config = (-2, 4000) #(4,4000)
    fusy_config = (-2, 4000) #(-3,4000)
    rt = report_type
#    if export:
#        csv_dic.to_csv("log_final/"+report_type+"_mkl.csv")
#        return 1
    if report_type == "MV-MV" or report_type == "MV-MV DIFF" or report_type == "MV-MV IND":
        sep1 = "Parallel Non-fused CSC-CSC"
        par1 = "Parallel Fused CSC-CSC"
        post_fix = "_CC.csv"
        rt_cc = rt + "-CSC-CSC"
        if lib_file:
            par1 = "Parallel MKL Non-fused CSC-CSC"
            post_fix = "_CC_mkl.csv"
            rt_cc += "-MKL"
        merged_df = speed_up_graph(csv_dic, out_path+rt_cc+'-speedup', sep1, par1, rt_cc)
        if export:
            global_dict["TYPE"] = rt_cc
            #merged_df.to_csv("log_final/" + report_type + post_fix)
            merged_df.to_csv(join(export_path,  report_type + post_fix))
            dic_to_csv(global_dict, join(out_path,"global_report.csv"))
        sep1 = "Parallel Non-fused CSR-CSR"
        par1 = "Parallel Fused CSR-CSR"
        post_fix = "_RR.csv"
        rt += "-CSR-CSR"
        if lib_file:
            par1 = "Parallel MKL Non-fused CSR-CSR"
            post_fix = "_RR_mkl.csv"
            rt += "-MKL"
        merged_df = speed_up_graph(csv_dic, out_path + rt + '-speedup', sep1, par1, rt)
        if export:
            global_dict["TYPE"] = rt
            #merged_df.to_csv("log_final/" + report_type + post_fix)
            merged_df.to_csv(join(export_path,  report_type + post_fix))
            dic_to_csv(global_dict, "global_report.csv")
            export = False
    elif report_type == "TRSV-MV":
        sep1 = "Parallel LBC Non-fused CSR-CSC"
        par1 = "Parallel Fused CSR-CSC NOR"
        #par1 = "Parallel Fused CSR-CSC"
        rt += "-CSR-CSC"
        if lib_file:
            par1 = "Parallel MKL Non-fused CSR-CSC"
            rt += "-MKL"
        #csv_dic_gp = extract_tuned(csv_dic)
        if not lib_file:
            csv_dic_gp1 = extract_tuned_tconfig(csv_dic, lbc_config)
            csv_dic_gp2 = extract_tuned_tconfig(csv_dic, fusy_config)
            merged_df = speed_up_graph_twogb(csv_dic_gp1, csv_dic_gp2, out_path+rt+'speedup',
                                             sep1, par1, rt)
        else:
            merged_df = speed_up_graph(csv_dic, out_path + rt + '-speedup', sep1, par1,
                                       rt)

    elif report_type == "MV-TRSV":
        sep1 = "Parallel LBC Non-fused CSC-CSC"
        par1 = "Parallel Fused CSC-CSC"
        rt_cc = rt + "-CSC-CSC"
        post_fix = "_CC.csv"
        if lib_file:
            par1 = "Parallel MKL Non-fused CSC-CSC"
            post_fix = "_CC_mkl.csv"
            rt_cc += "-MKL"
        #csv_dic_gp = extract_tuned(csv_dic)
        if not lib_file:
            csv_dic_gp1 = extract_tuned_tconfig(csv_dic, lbc_config)
            csv_dic_gp2 = extract_tuned_tconfig(csv_dic, fusy_config)
            merged_df = speed_up_graph_twogb(csv_dic_gp1, csv_dic_gp2, out_path + rt_cc +
                                             '-speedup', sep1, par1, rt_cc)
        else:
            merged_df = speed_up_graph(csv_dic, out_path + rt_cc+'speedup', sep1, par1,
                                       rt_cc)
        if export:
            global_dict["TYPE"] = rt_cc
            #merged_df.to_csv("log_final/" + report_type + post_fix)
            #dic_to_csv(global_dict, "global_report.csv")
            merged_df.to_csv(join(export_path,  report_type + post_fix))
            dic_to_csv(global_dict, join(out_path,"global_report.csv"))

        sep1 = "Parallel LBC Non-fused CSR-CSR"
        par1 = "Parallel Fused CSR-CSR"
        post_fix = "_RR.csv"
        rt += "-CSR-CSR"
        if lib_file:
            par1 = "Parallel MKL Non-fused CSR-CSR"
            post_fix = "_RR_mkl.csv"
            rt += "-MKL"
        if not lib_file:
            csv_dic_gp1 = extract_tuned_tconfig(csv_dic, lbc_config)
            csv_dic_gp2 = extract_tuned_tconfig(csv_dic, fusy_config)
            merged_df = speed_up_graph_twogb(csv_dic_gp1, csv_dic_gp2, out_path + rt + '-speedup', sep1, par1, rt)
        else:
            merged_df = speed_up_graph(csv_dic, out_path+rt+'-speedup', sep1, par1, rt)
        if export:
            global_dict["TYPE"] = rt
            #merged_df.to_csv("log_final/" + report_type + post_fix)
            #dic_to_csv(global_dict, "global_report.csv")
            merged_df.to_csv(join(export_path,  report_type + post_fix))
            dic_to_csv(global_dict, join(out_path,"global_report.csv"))
            export = False

    elif report_type == "IC0-TRSV":
        sep1 = "Parallel LBC Non-fused CSC-CSC"
        par1 = "Parallel Fused Code CSC-CSC TW"
        rt += "-CSR-CSR"
        #csv_dic_gp = extract_tuned(csv_dic)
        #speed_up_graph(csv_dic_gp, out_path + report_type + 'csr-csr-speedup', sep1, par1, report_type + ' CSR-CSR')
        if not lib_file:
            csv_dic_gp1 = extract_tuned_tconfig(csv_dic, lbc_config)
            csv_dic_gp2 = extract_tuned_tconfig(csv_dic, fusy_config)
            merged_df = speed_up_graph_twogb(csv_dic_gp1, csv_dic_gp2, out_path + rt + '-speedup', sep1, par1, rt)
        else:
            rt += "-LIB"
            par1 = "Parallel LIB Non-fused CSR-CSR"
            merged_df = speed_up_graph(csv_dic, out_path+rt+'-speedup', sep1, par1, rt)
    elif report_type == "DAD-IC0":
        sep1 = "Parallel LBC Non-fused CSR-CSR"
        par1 = "Parallel Fused Code CSR-CSR"
        rt += "-CSR-CSR"
        #csv_dic_gp = extract_tuned(csv_dic)
        #speed_up_graph(csv_dic_gp, out_path + report_type + 'csr-csr-speedup', sep1, par1, report_type + ' CSR-CSR')
        if not lib_file:
            csv_dic_gp1 = extract_tuned_tconfig(csv_dic, lbc_config)
            csv_dic_gp2 = extract_tuned_tconfig(csv_dic, fusy_config)
            merged_df = speed_up_graph_twogb(csv_dic_gp1, csv_dic_gp2, out_path + rt + '-speedup', sep1, par1, rt)
        else:
            rt += "-LIB"
            par1 = "Parallel LIB Non-fused CSR-CSR"
            merged_df = speed_up_graph(csv_dic, out_path+rt+'-speedup', sep1, par1, rt)
        #speed_up_graph(csv_dic, out_path+report_type+'csr-csr-speedup', sep1, par1, report_type+' CSR-CSR')
    elif report_type == "ILU0-TRSV":
        sep1 = "Parallel LBC Non-fused CSR-CSR"
        par1 = "Parallel Fused Code CSR-CSR"
        rt += "-CSR-CSR"
        #csv_dic_gp = extract_tuned(csv_dic)
        #speed_up_graph(csv_dic_gp, out_path + report_type + 'csr-csr-speedup', sep1, par1, report_type + ' CSR-CSR')
        if not lib_file:
            csv_dic_gp1 = extract_tuned_tconfig(csv_dic, lbc_config)
            csv_dic_gp2 = extract_tuned_tconfig(csv_dic, fusy_config)
            merged_df = speed_up_graph_twogb(csv_dic_gp1, csv_dic_gp2, out_path + rt + '-speedup', sep1, par1, rt)
        else:
            rt += "-MKL"
            par1 = "Parallel MKL Non-fused CSR-CSR"
            merged_df = speed_up_graph(csv_dic, out_path+rt+'-speedup', sep1, par1, rt)
    elif report_type == "DAD-ILU0":
        sep1 = "Parallel LBC Non-fused CSR-CSR"
        par1 = "Parallel Fused Code CSR-CSR"
        rt += "-CSR-CSR"
        #csv_dic_gp = extract_tuned(csv_dic)
        #speed_up_graph(csv_dic_gp, out_path + report_type + 'csr-csr-speedup', sep1, par1, report_type + ' CSR-CSR')
        if not lib_file:
            csv_dic_gp1 = extract_tuned_tconfig(csv_dic, lbc_config)
            csv_dic_gp2 = extract_tuned_tconfig(csv_dic, fusy_config)
            merged_df = speed_up_graph_twogb(csv_dic_gp1, csv_dic_gp2, out_path + rt + '-speedup', sep1, par1, rt)
        else:
            rt += "-MKL"
            par1 = "Parallel MKL Non-fused CSR-CSR"
            merged_df = speed_up_graph(csv_dic, out_path+rt+'-speedup', sep1, par1, rt)
    elif report_type == "TRSV-TRSV":
        sep1 = "Parallel LBC Non-fused CSR-CSR"
        par1 = "Parallel Fused CSR-CSR"
        rt += "-CSR-CSR"
        #csv_dic_gp = extract_tuned(csv_dic)
        #speed_up_graph(csv_dic_gp, out_path + report_type + 'csr-csr-speedup', sep1, par1, report_type + ' CSR-CSR')
        if not lib_file:
            csv_dic_gp1 = extract_tuned_tconfig(csv_dic, lbc_config)
            csv_dic_gp2 = extract_tuned_tconfig(csv_dic, fusy_config)
            merged_df = speed_up_graph_twogb(csv_dic_gp1, csv_dic_gp2, out_path + rt + '-speedup', sep1, par1, rt)
        else:
            rt += "-MKL"
            par1 = "Parallel MKL Non-fused CSR-CSR"
            merged_df = speed_up_graph(csv_dic, out_path+rt+'-speedup', sep1, par1, rt)

    elif report_type == "LDL-TRSV":
        sep1 = "Parallel LBC Non-fused CSC-CSC"
        par1 = "Parallel Fused Code CSC-CSC"
        rt += "-CSC-CSC"
        #csv_dic_gp = extract_tuned(csv_dic)
        #speed_up_graph(csv_dic_gp, out_path + report_type + 'csr-csr-speedup', sep1, par1, report_type + ' CSR-CSR')
        if not lib_file:
            csv_dic_gp1 = extract_tuned_tconfig(csv_dic, lbc_config)
            csv_dic_gp2 = extract_tuned_tconfig(csv_dic, fusy_config)
            merged_df = speed_up_graph_twogb(csv_dic_gp1, csv_dic_gp2, out_path + rt + '-speedup', sep1, par1, rt)
        else:
            rt += "-LIB"
            par1 = "Parallel LIB Non-fused CSC-CSC"
            merged_df = speed_up_graph(csv_dic, out_path+rt+'-speedup', sep1, par1, rt)

    elif report_type == "TRSV-MV-PROF":
        par1 = "Parallel Fused CSR-CSC"
        par2 = "Parallel Fused CSR-CSC Atomic Interleaved"
        [t1, t2] = compute_locality_ratio_by_variant_matrix(csv_dic, par1, par2)
        b, m = polyfit(t1, t2, 1)
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)
        ax1.scatter(t1, t2)
        ax1.plot(t1, b + m * t1, '-')
        plt.axis()
        ax1.set_xlabel("Interleaved / Separated Memory Access Time", fontsize=14)
        ax1.set_ylabel("Separated / Interleaved Locality Cost", fontsize=14)
        print( "cor: ", np.corrcoef(t1, t2), sep='=')
        fig.savefig(report_type + ".pdf", bbox_inches='tight')
        plt.show()
        #plot_scatter_chart(np.concatenate(t1, t2), )
        #speed_up_graph(csv_dic_gp, out_path+report_type+'csr-csc-speedup', sep1, par1, report_type+' CSR-CSC')
    else:
        print("Type is not supported.\n")
        return 0
    global_dict["TYPE"] = rt
    if export:
        if lib_file:
            #merged_df.to_csv("log_final/"+report_type+"_mkl.csv")
            merged_df.to_csv(join(export_path,  report_type + "_mkl.csv"))
        else:
            #merged_df.to_csv("log_final/"+report_type+".csv")
            merged_df.to_csv(join(export_path,  report_type + ".csv"))
        #dic_to_csv(global_dict, "global_report.csv")
        dic_to_csv(global_dict, join(out_path,"global_report.csv"))

    print("==> plotted", rt)
    return 1

