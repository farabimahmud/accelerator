import sys
import numpy as np
import re
import json
import matplotlib.pyplot as plt
from scipy import stats
from easypyplot import barchart, color, pdf
from easypyplot import format as fmt
from copy import deepcopy
import os

def add_line(ax, xpos, ypos):
    line = plt.Line2D(
        #[xpos, xpos], [ypos + linelen, ypos],
        [xpos, xpos],
        [0, ypos],
        transform=ax.transAxes,
        color='black',
        linewidth=1)
    line.set_clip_on(False)
    ax.add_line(line)


def main(folder_path):

    benchmarks = ['alexnet', 'AlphaGoZero', 'FasterRCNN', 'Googlenet', 'NCF_recommendation',
            'Resnet50', 'Transformer']
    names = ['ring', 'mxnettree_alpha', 'mxnettree_beta', 'multitree_alpha', 'multitree_beta', 'multitree_gamma']
    schemes = ['Ring', 'MXNetTree-$\\alpha$', 'MXNettree-$\\beta$', 'MultiTree-$\\alpha$', 'MultiTree-$\\beta$', 'MultiTree-$\\gamma$']

    entry_names = ['Allreduce', 'Training']
    energy_entry_names = ['Dynamic', 'Static']
    xlabels = ['AlexNet','AlphaGoZero', 'FasterRCNN', 'GoogleNet', 'NCF-recommendation', 'Resnet50', 'Transformer']
    xlabels.append('gmean')
    group_names = []

    cycles = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    energy_cycles = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    norm_cycles = np.zeros(
        (int(len(schemes)), int(len(xlabels))), dtype=np.float)
    norm_allreduce_cycles = np.zeros(
        (int(len(schemes)), int(len(xlabels))), dtype=np.float)
    training_cycles = np.zeros((int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    allreduce_cycles = np.zeros((int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    cycles_breakdown = np.zeros((2, int(len(benchmarks) * len(schemes))), dtype=np.float)
    norm_cycles_breakdown = np.zeros((2, int(len(benchmarks) * len(schemes))), dtype=np.float)

    total_power = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    dynamic_power = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    static_power = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    power_breakdown = np.zeros((2, int(len(benchmarks) * len(schemes))), dtype=np.float)
    norm_power_breakdown = np.zeros((2, int(len(benchmarks) * len(schemes))), dtype=np.float)
    total_energy = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    dynamic_energy = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    static_energy = np.zeros(
        (int(len(schemes)), int(len(benchmarks))), dtype=np.float)
    energy_breakdown = np.zeros((2, int(len(benchmarks) * len(schemes))), dtype=np.float)
    norm_energy_breakdown = np.zeros((2, int(len(benchmarks) * len(schemes))), dtype=np.float)

    for s, scheme in enumerate(schemes):
        for b, bench in enumerate(benchmarks):
            filename = folder_path + '/' + bench + '_' + names[s] + '.json'

            with open(filename, 'r') as json_file:
                sim = json.load(json_file)

                allreduce_cycles[s][b] = sim['results']['performance']['allreduce']['total']
                training_cycles[s][b] = sim['results']['performance']['training']
                if training_cycles[s][b] != 0:
                    energy_cycles[s][b] = sim['results']['performance']['total']
                else:
                    energy_cycles[s][b] = allreduce_cycles[s][b]
                if scheme == 'Ring':
                    cycles[s][b] = sim['results']['performance']['total']
                else:
                    training_cycles[s][b] = training_cycles[0][b]
                    cycles[s][b] = allreduce_cycles[s][b] + training_cycles[s][b]

                norm_cycles[s][b] = cycles[s][b] / cycles[0][b]
                norm_allreduce_cycles[s][b] = allreduce_cycles[s][b] / allreduce_cycles[0][b]
                cycles_breakdown[0][b * len(schemes) + s] = allreduce_cycles[s][b]
                cycles_breakdown[1][b * len(schemes) + s] = training_cycles[s][b]

                total_power[s][b] = sim['results']['power']['network']['total']
                dynamic_power[s][b] = sim['results']['power']['network']['dynamic']
                static_power[s][b] = sim['results']['power']['network']['static']
                power_breakdown[0][b * len(schemes) + s] = dynamic_power[s][b]
                power_breakdown[1][b * len(schemes) + s] = static_power[s][b]

                dynamic_energy[s][b] = dynamic_power[s][b] * energy_cycles[s][b]
                static_energy[s][b] = static_power[s][b] * cycles[s][b]
                total_energy[s][b] = dynamic_energy[s][b] + static_energy[s][b]
                energy_breakdown[0][b * len(schemes) + s] = dynamic_energy[s][b]
                energy_breakdown[1][b * len(schemes) + s] = static_energy[s][b]

                json_file.close()

        norm_cycles[s][-1] = stats.mstats.gmean(norm_cycles[s][0:-1])
        norm_allreduce_cycles[s][-1] = stats.mstats.gmean(norm_allreduce_cycles[s][0:-1])

    speedup = 1.0 / norm_cycles
    allreduce_speedup = 1.0 / norm_allreduce_cycles
    speedup[np.isnan(speedup)] = 0
    allreduce_speedup[np.isnan(allreduce_speedup)] = 0

    for b, bench in enumerate(benchmarks):
        for s, scheme in enumerate(schemes):
            group_names.append(scheme)
            for e, entry in enumerate(entry_names):
                norm_cycles_breakdown[e][b * len(schemes) + s] = cycles_breakdown[e][b * len(schemes) + s] / cycles[0][b]
    norm_cycles_breakdown[np.isnan(norm_cycles_breakdown)] = 0
    for b, bench in enumerate(benchmarks):
        for s, scheme in enumerate(schemes):
            for e, entry in enumerate(energy_entry_names):
                norm_energy_breakdown[e][b * len(schemes) + s] = energy_breakdown[e][b * len(schemes) + s] / total_energy[0][b]
                norm_energy_breakdown[e][b * len(schemes) + s] = energy_breakdown[e][b * len(schemes) + s] / total_energy[0][b]
    norm_power_breakdown[np.isnan(norm_power_breakdown)] = 0
    norm_energy_breakdown[np.isnan(norm_energy_breakdown)] = 0
    #print(norm_cycles_breakdown)

    '''
    result_file = open('performance.csv', mode='w')
    result_file.write('cycles:\n')
    head = ''
    for b, bench in enumerate(benchmarks):
        head += ',{}'.format(bench)
    head += '\n'
    result_file.write(head)
    for s, scheme in enumerate(schemes):
        row = scheme
        for b, benchmark in enumerate(benchmarks):
            row = row + ',' + str(cycles[s][b])
        row = row + '\n'
        result_file.write(row)
    result_file.write('\n')
    '''

    colors = ['#e0f3db','#a8ddb5','#43a2ca', '#e0f3db','#a8ddb5','#43a2ca']
    colors = ['#f0f9e8','#ccebc5','#a8ddb5','#7bccc4','#43a2ca','#0868ac']
    plt.rc('legend', fontsize=22)
    plt.rc('font', size=18)

    data = [list(i) for i in zip(*speedup)]
    data = np.array(data, dtype=np.float64)
    #fig = plt.figure(figsize=(8, 5.5))
    figpath = folder_path + '/speedup.pdf'
    pdfpage, fig = pdf.plot_setup(figpath, figsize=(8, 5), fontsize=22, font=('family', 'Tw Cen MT'))
    ax = fig.gca()
    hdls = barchart.draw(
        ax,
        data,
        group_names=xlabels,
        entry_names=schemes,
        colors=colors,
        breakdown=False,
        legendloc='upper center',
        legendncol=len(schemes))
    fig.autofmt_xdate()
    #ax.set_ylim(0, 20)
    ax.yaxis.grid(True, linestyle='--')
    ax.set_ylabel('Runtime Speedup')
    ax.legend(
        hdls,
        schemes,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=len(schemes),
        frameon=False,
        handletextpad=0.6,
        columnspacing=1)
    fmt.resize_ax_box(ax, hratio=0.8)
    pdf.plot_teardown(pdfpage)
    ##############################
    # normalized runtime breakdown
    ##############################
    #colors = ['#8faadc', '#e2f0d9', '#f4b183']
    colors = ['#2b8cbe', '#e2f0d9', '#f4b183']
    xticks = []
    for i in range(0, len(benchmarks)):
        for j in range(0, len(schemes)):
            xticks.append(i * (len(schemes) + 1) + j)
    data = [list(i) for i in zip(*norm_cycles_breakdown)]
    data = np.array(data, dtype=np.float64)
    figpath = folder_path + '/norm_runtime.pdf'
    pdfpage, fig = pdf.plot_setup(figpath, figsize=(30, 8), fontsize=26, font=('family', 'Tw Cen MT'))
    ax = fig.gca()
    ax2 = ax.twinx()  # ax for allreduce speedup
    hdls = barchart.draw(
        ax,
        data,
        group_names=group_names,
        entry_names=entry_names,
        breakdown=True,
        xticks=xticks,
        width=0.8,
        colors=colors,
        legendloc='upper center',
        legendncol=len(entry_names),
        xticklabelfontsize=22,
        xticklabelrotation=90,
        log=False)
    ax.set_ylabel('Normalized Runtime Breakdown')
    ax.yaxis.grid(True, linestyle='--')
    fmt.resize_ax_box(ax, hratio=0.78)
    ly = len(benchmarks)
    scale = 1. / ly
    ypos = -.41
    pos = 0
    for pos in range(ly + 1):
        lxpos = (pos + 0.5) * scale
        if pos < ly:
            ax.text(
                lxpos, ypos, xlabels[pos], ha='center', transform=ax.transAxes)
        add_line(ax, pos * scale, ypos)



    #############################
    # All reduce speedup
    #############################
    #[       [1.         0.4994861  1.32979743 0.6254262  1.9101432  2.5267156 ]
    #        [1.         0.51647551 1.39716977 0.61125715 1.87302411 2.5307805 ]
    #        [1.         0.53101313 1.37975687 0.59869921 1.9078746  2.524382  ]
    #        [1.         0.53116167 1.27836285 0.60494638 1.93217098 2.52539887]
    #        [1.         0.4747683  1.38020831 0.61143355 1.90264861 2.52471041]
    #        [1.         0.53600642 1.23278946 0.62154002 1.85625354 2.52383927]
    #        [1.         0.55971783 1.34595079 0.60816199 1.95166229 2.52353277]
    #        [1.         0.52059946 1.33367677 0.61157805 1.9045867  2.52562171]]

    # generate x position for allreduce
    xs = []
    p = 0.0
    for g in range(7):
        xs.append([])
        for pos in range(6):
            xs[g].append(p)
            p = p + 1
        p = p + 1

    data = [list(i) for i in zip(*allreduce_speedup)]
    data = np.array(data, dtype=np.float64)
    ax2.set_ylim(0, 3)
    ax2.set_ylabel('Allreduce Speedup')
    for i in range(7):
        tmp = ax2.plot(xs[i], data[i], '-o', markersize=8, color='black', markeredgecolor='#4b4a25')
        if i == 0:
            hdls += tmp
    ax.legend(
        hdls,
        entry_names + ['Allreduce Speedup'],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        #bbox_to_anchor=(0.5, 1.1),
        ncol=len(entry_names)+1,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1)

    pdf.plot_teardown(pdfpage)

    # normalized power breakdown
    colors = ['#a63603','#fee6ce']
    xticks = []
    for i in range(0, len(benchmarks)):
        for j in range(0, len(schemes)):
            xticks.append(i * (len(schemes) + 1) + j)
    data = [list(i) for i in zip(*norm_energy_breakdown)]
    data = np.array(data, dtype=np.float64)
    figpath = folder_path + '/norm_energy.pdf'
    pdfpage, fig = pdf.plot_setup(figpath, figsize=(30, 8), fontsize=26, font=('family', 'Tw Cen MT'))
    ax = fig.gca()
    hdls = barchart.draw(
        ax,
        data,
        group_names=group_names,
        entry_names=energy_entry_names,
        breakdown=True,
        xticks=xticks,
        width=0.8,
        colors=colors,
        legendloc='upper center',
        legendncol=len(energy_entry_names),
        xticklabelfontsize=22,
        xticklabelrotation=90,
        log=False)
    ax.set_ylabel('Normalized Energy Breakdown')
    ax.yaxis.grid(True, linestyle='--')
    ax.legend(
        hdls,
        energy_entry_names,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(energy_entry_names),
        frameon=False,
        handletextpad=0.6,
        columnspacing=1)
    fmt.resize_ax_box(ax, hratio=0.78)
    ly = len(benchmarks)
    scale = 1. / ly
    ypos = -.41
    pos = 0
    for pos in range(ly + 1):
        lxpos = (pos + 0.5) * scale
        if pos < ly:
            ax.text(
                lxpos, ypos, xlabels[pos], ha='center', transform=ax.transAxes)
        add_line(ax, pos * scale, ypos)
    pdf.plot_teardown(pdfpage)

    plt.show()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('usage: ' + sys.argv[0] + ' folder_path')
        exit()
    main(sys.argv[1])
