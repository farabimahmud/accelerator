import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from easypyplot import pdf, barchart, color
from easypyplot import format as fmt

def main(folder_path):

    schemes = ['Ring', 'MXNetTree', 'MultiTree']

    algorithmic_scalability = {}

    alg_scale_file = folder_path + '/algorithmic_scalability.csv'

    with open(alg_scale_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            algorithmic_scalability[row[0]] = [int(ele) for ele in row[1:]]

    csvfile.close()

    plt.rc('legend', fontsize=18)
    plt.rc('font', size=18)

    colors = ['#edb120','#d95319','#0071bd']
    makercolors = ['#f7dea3','#fcc4ac','#addaf7']
    linestyles = ['-', '-', '-']
    markers = ['o', '^', 's']

    figname = folder_path + '/algorithmic_scalability.pdf'
    pdfpage, fig = pdf.plot_setup(figname, figsize=(8, 5), fontsize=18)
    ax = fig.gca()
    for s, scheme in enumerate(schemes):
        ax.plot(
                algorithmic_scalability['nodes'],
                algorithmic_scalability[scheme],
                marker=markers[s],
                markersize=10,
                markeredgecolor=colors[s],
                markerfacecolor=makercolors[s],
                markeredgewidth=2,
                color=colors[s],
                linestyle=linestyles[s],
                linewidth=2,
                label=scheme
                )
        ax.set_xticks(algorithmic_scalability['nodes'])
        ax.set_xlim(0, 270)
        ax.set_xlabel('Number of Nodes in 2D Torus Network')
        ax.set_ylabel('Algorithmic Time Steps')
        ax.yaxis.grid(True, linestyle='--', color='black')
        hdls, lab = ax.get_legend_handles_labels()
        ax.legend(
                hdls,
                lab,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.2),
                ncol=3,
                frameon=False,
                handletextpad=0.6,
                columnspacing=1
                )
        fig.subplots_adjust(top=0.8, bottom=0.2)
    pdf.plot_teardown(pdfpage, fig)

    plt.show()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('usage: ' + sys.argv[0] + ' folder_path')
        exit()
    main(sys.argv[1])
