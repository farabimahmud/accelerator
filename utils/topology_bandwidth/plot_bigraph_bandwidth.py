#!/bin/python3.6
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from easypyplot import pdf, barchart, color
from easypyplot import format as fmt
from collections import defaultdict
import json

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

def main(folder_path):
    # ring, multitree-alph, multitree-beta, multitree-gamma, hdm
    schemes = ['Ring', 'MultiTree-$\\alpha$', 'MultiTree-$\\gamma$', 'Hdrm']
    names = ['ring', 'multitree_alpha', 'multitree_gamma', 'hdrm']

    node = 32
    ldata = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    #xlabels = ['32', '', '', '256', '', '', '2,048', '', '', '16,384', '', '']
    xlabels = ['', '64', '', '', '512', '', '', '4,096', '', '', '32,768', '']
    xlabels = ['', '64KB', '', '', '512KB', '', '', '4MB', '', '', '32MB', '']
    elements = [4]
    #elements = [4, 6]

    # algorithmic_scalability was here

    gbps = {}
    comm_cycles = {}

    #get the file names
    for s, name in enumerate(names):
        if name not in comm_cycles.keys():
            comm_cycles[name] = {}

        for e, element in enumerate(elements):
            if element not in comm_cycles[name].keys():
                comm_cycles[name][element] = {}

            for d, data in enumerate(ldata):
                if data not in comm_cycles[name][element].keys():
                    comm_cycles[name][element][data] = []

                #data = 98304*node
                filename = "%s/%dnodes_%dkB_%delementsize_%s.json" % (folder_path, node, data, element, name)
                print (filename)
                with open(filename, 'r') as json_file:
                    sim = json.load(json_file)
                    comm_cycles[name][element][data] = float(sim['results']['performance']['allreduce']['communication'])

    for s, name in enumerate(names):
        if name not in gbps.keys():
            gbps[name] = {}

        for e, element in enumerate(elements):
            if element not in gbps[name].keys():
                gbps[name][element] = []

            for d, data in enumerate(ldata):
                #gbps[name][element].append( (2*(node-1)*(data/(1024*1024))) / (comm_cycles[name][element][data] / (10 ** 9) ))
                gbps[name][element].append( ((float(data)/(1024*1024))) / (comm_cycles[name][element][data] / (10 ** 9) ))

    for s, scheme in enumerate(names):
        for e, element in enumerate(elements):
            print ("%s-%s" % (scheme, element))
            print (gbps[scheme][element])


    plt.rc('legend', fontsize=18)
    plt.rc('font', size=18)



    # matlab color palette
    colors = ['#edb120','#d95319','#0071bd','#0071bd','#0071bd','#0071bd']
    makercolors = ['#f7dea3','#fcc4ac','#addaf7','#addaf7','#addaf7','#addaf7']
    # powerpoint color palette
    colors = ['#70ad47','#ed7d31','#4472c4','#a63603','#4472c4','#4472c4']
    makercolors = ['#e2f0d9','#fbe5d6','#dae3f3','#fee6ce','#cccccc','#dae3f3']
    linestyles = ['-', '-', '-', '-', '-', '-', '-']
    markers = ['o', '^', 's', 'D', 's', 's']

    figname = './bigraph_bandwidth.pdf'
    pdfpage, fig = pdf.plot_setup(figname, figsize=(8, 6), fontsize=22, font=('family', 'Tw Cen MT'))
    ax = fig.gca()
    for s, scheme in enumerate(names):
        for e, element in enumerate(elements):
            ax.plot(
                    gbps[scheme][element],
                    marker=markers[s],
                    markersize=14,
                    markeredgecolor=colors[s],
                    markerfacecolor=makercolors[s],
                    markeredgewidth=3,
                    color=colors[s],
                    linestyle=linestyles[s],
                    linewidth=3,
                    label=schemes[s],
                    )
            ax.set_xticks(range(len(ldata)))
            #ax.set_xticklabels(ldata_legend,
            #        rotation = 45, ha="center",)
            ax.set_xticklabels(xlabels)
            #ax.set_xlim(0, 270)
            ax.set_ylim(0, 10)
            ax.set_xlabel('All-Reduce Data Size')
            ax.set_ylabel('Bandwidth (GB/s)')
            ax.yaxis.grid(True, linestyle='--', color='black')
            hdls, lab = ax.get_legend_handles_labels()
            legend = ax.legend(
                    hdls,
                    lab,
                    loc='lower right',
                    #bbox_to_anchor=(0.5, 1.25),
                    ncol=1,
                    #frameon=False,
                    frameon=True,
                    handletextpad=0.6,
                    columnspacing=1
                    )
            legend.get_frame().set_edgecolor('white')
            #fig.subplots_adjust(top=0.8, bottom=0.2)
            fig.subplots_adjust(top=0.95, bottom=0.15)
    pdf.plot_teardown(pdfpage, fig)

    plt.show()

if __name__== "__main__":
    if len(sys.argv) != 2:
        print('usage: ' + sys.argv[0] + ' folder_path')
        exit()
    main(sys.argv[1])
