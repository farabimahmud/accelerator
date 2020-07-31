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
    schemes = ['Ring', 'MultiTree-$\\beta$', 'MultiTree-$\\delta$']
    names = ['ring', 'multitree_beta', 'multitree_delta']

    node = 16
    ldata = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
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
                    comm_cycles[name][element][data] = sim['results']['performance']['allreduce']['communication']

    for s, name in enumerate(names):
        if name not in gbps.keys():
            gbps[name] = {}

        for e, element in enumerate(elements):
            if element not in gbps[name].keys():
                gbps[name][element] = []

            for d, data in enumerate(ldata):
                #gbps[name][element].append( (2*(node-1)*(data/(1024*1024))) / (comm_cycles[name][element][data] / (10 ** 9) ))
                gbps[name][element].append( ((data/(1024*1024))) / (comm_cycles[name][element][data] / (10 ** 9) ))

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
    colors = ['#70ad47','#ed7d31','#4472c4','#4472c4','#4472c4','#4472c4']
    makercolors = ['#e2f0d9','#fbe5d6','#dae3f3','#dae3f3','#dae3f3','#dae3f3']
    linestyles = ['-', '-', '-', '-', '-', '-', '-']
    markers = ['o', '^', 's', 's', 's', 's']

    figname = './plot_torus_bandwidth.pdf'
    pdfpage, fig = pdf.plot_setup(figname, figsize=(8, 5), fontsize=22, font=('family', 'Tw Cen MT'))
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
                    label=scheme+str(element)
                    )
            ax.set_xticks(range(len(ldata)))
            ax.set_xticklabels(ldata)
            #ax.set_xlim(0, 270)
            #ax.set_ylim(0, 18)
            ax.set_xlabel('Data Size')
            ax.set_ylabel('GBps')
            ax.yaxis.grid(True, linestyle='--', color='black')
            hdls, lab = ax.get_legend_handles_labels()
            ax.legend(
                    hdls,
                    lab,
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.25),
                    ncol=3,
                    frameon=False,
                    handletextpad=0.6,
                    columnspacing=1
                    )
            fig.subplots_adjust(top=0.8, bottom=0.2)
    pdf.plot_teardown(pdfpage, fig)

    plt.show()

if __name__== "__main__":
    if len(sys.argv) != 2:
        print('usage: ' + sys.argv[0] + ' folder_path')
        exit()
    main(sys.argv[1])
