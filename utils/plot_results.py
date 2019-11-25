import sys
import numpy as np
import re
import json
import matplotlib.pyplot as plt
from scipy import stats
from easypyplot import barchart, color, pdf
from easypyplot import format as fmt
from copy import deepcopy


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

    #benchmarks = ['AlphaGoZero', 'DeepSpeech2', 'FasterRCNN', 'NCF_recommendation',
    #        'NCF_recommendation_short', 'Resnet50', 'Sentimental_seqCNN',
    #        'Sentimental_seqLSTM', 'Sentimental_seqLSTM_short', 'Transformer',
    #        'Transformer_short']
    benchmarks = ['AlphaGoZero', 'FasterRCNN', 'NCF_recommendation', 'Resnet50', 'Transformer_short']
    #benchmarks = ['AlphaGoZero', 'FasterRCNN', 'NCF_recommendation', 'Transformer_short']
    #benchmarks = ['AlphaGoZero', 'NCF_recommendation_short', 'Transformer_short']
    names = ['ring', 'mxnettree', 'multitree']
    schemes = ['Ring', 'MXNetTree', 'MultiTree']

    entry_names = ['Allreduce', 'Training']
    energy_entry_names = ['Dynamic', 'Static']
    xlabels = ['AlphaGoZero', 'FasterRCNN', 'NCF', 'Resnet50', 'Transformer-s']
    #xlabels = ['AlphaGoZero', 'FasterRCNN', 'NCF', 'Transformer-s']
    #xlabels = ['AlphaGoZero', 'NCF-s', 'Transformer-s']
    xlabels.append('gmean')
    group_names = []

    cycles = np.zeros(
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

                cycles[s][b] = sim['results']['performance']['total']
                training_cycles[s][b] = sim['results']['performance']['training']
                allreduce_cycles[s][b] = sim['results']['performance']['allreduce']['total']

                norm_cycles[s][b] = cycles[s][b] / cycles[0][b]
                norm_allreduce_cycles[s][b] = allreduce_cycles[s][b] / allreduce_cycles[0][b]
                cycles_breakdown[0][b * len(schemes) + s] = allreduce_cycles[s][b]
                cycles_breakdown[1][b * len(schemes) + s] = training_cycles[s][b]

                total_power[s][b] = sim['results']['power']['network']['total']
                dynamic_power[s][b] = sim['results']['power']['network']['dynamic']
                static_power[s][b] = sim['results']['power']['network']['static']
                power_breakdown[0][b * len(schemes) + s] = dynamic_power[s][b]
                power_breakdown[1][b * len(schemes) + s] = static_power[s][b]

                total_energy[s][b] = total_power[s][b] * cycles[s][b]
                dynamic_energy[s][b] = dynamic_power[s][b] * cycles[s][b]
                static_energy[s][b] = static_power[s][b] * cycles[s][b]
                energy_breakdown[0][b * len(schemes) + s] = dynamic_energy[s][b]
                energy_breakdown[1][b * len(schemes) + s] = static_energy[s][b]

                json_file.close()

        norm_cycles[s][-1] = stats.mstats.gmean(norm_cycles[s][0:-1])
        norm_allreduce_cycles[s][-1] = stats.mstats.gmean(norm_allreduce_cycles[s][0:-1])

    speedup = 1.0 / norm_cycles
    allreduce_speedup = 1.0 / norm_allreduce_cycles
    speedup[np.isnan(speedup)] = 0
    allreduce_speedup[np.isnan(allreduce_speedup)] = 0
    #speedup[speedup == np.inf] = 0
    #speedup[speedup == -np.inf] = 0
    #allreduce_speedup[allreduce_speedup == np.inf] = 0
    #allreduce_speedup[allreduce_speedup == -np.inf] = 0
    #print(speedup)
    #print(allreduce_speedup)

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

    colors = ['#e0f3db','#a8ddb5','#43a2ca']
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

    data = [list(i) for i in zip(*allreduce_speedup)]
    data = np.array(data, dtype=np.float64)
    #fig = plt.figure(figsize=(8, 5.5))
    figpath = folder_path + '/allreduce_speedup.pdf'
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
    ax.set_ylabel('Allreduce Speedup')
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

    # normalized runtime breakdown
    #colors = ['#8faadc', '#e2f0d9', '#f4b183']
    colors = ['#2b8cbe', '#e2f0d9', '#f4b183']
    xticks = []
    for i in range(0, len(benchmarks)):
        for j in range(0, len(schemes)):
            xticks.append(i * (len(schemes) + 1) + j)
    data = [list(i) for i in zip(*norm_cycles_breakdown)]
    data = np.array(data, dtype=np.float64)
    figpath = folder_path + '/norm_runtime.pdf'
    pdfpage, fig = pdf.plot_setup(figpath, figsize=(10, 6), fontsize=22, font=('family', 'Tw Cen MT'))
    ax = fig.gca()
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
        xticklabelfontsize=20,
        xticklabelrotation=90,
        log=False)
    ax.set_ylabel('Normalized Runtime Breakdown')
    ax.yaxis.grid(True, linestyle='--')
    ax.legend(
        hdls,
        entry_names,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(entry_names),
        frameon=False,
        handletextpad=0.6,
        columnspacing=1)
    fmt.resize_ax_box(ax, hratio=0.78)
    ly = len(benchmarks)
    scale = 1. / ly
    ypos = -.4
    pos = 0
    for pos in xrange(ly + 1):
        lxpos = (pos + 0.5) * scale
        if pos < ly:
            ax.text(
                lxpos, ypos, xlabels[pos], ha='center', transform=ax.transAxes)
        add_line(ax, pos * scale, ypos)
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
    pdfpage, fig = pdf.plot_setup(figpath, figsize=(10, 6), fontsize=22, font=('family', 'Tw Cen MT'))
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
        xticklabelfontsize=20,
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
    ypos = -.4
    pos = 0
    for pos in xrange(ly + 1):
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
