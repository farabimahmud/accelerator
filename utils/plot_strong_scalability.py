import sys
import numpy as np
import matplotlib.pyplot as plt
from easypyplot import pdf, barchart, color
from easypyplot import format as fmt
import json

def main(folder_path):

    schemes = ['Ring', 'MXNetTree-$\\beta$', 'MultiTree-$\\delta$']
    names = ['ring', 'mxnettree_beta', 'multitree_gamma']

    nodes = [16, 36, 64, 100, 144, 196, 256]

    algorithmic_scalability = {}

    cycles = np.zeros(
        (int(len(schemes)), int(len(nodes))), dtype=np.float)

    for s, name in enumerate(names):
        for n, node in enumerate(nodes):
            filename = folder_path + '/' + str(node) + 'nodes' + '_' + '8388608' + 'data' + '_' + name + '.json'
            print (filename)
            with open(filename, 'r') as json_file:
                sim = json.load(json_file)
                cycles[s][n] = sim['results']['performance']['total']

        algorithmic_scalability[name] = [int(ele)/cycles[0][0] for ele in cycles[s]]

    print(algorithmic_scalability)

    plt.rc('legend', fontsize=18)
    plt.rc('font', size=18)

    # matlab color palette
    colors = ['#edb120','#d95319','#0071bd']
    makercolors = ['#f7dea3','#fcc4ac','#addaf7']
    # powerpoint color palette
    colors = ['#70ad47','#ed7d31','#4472c4']
    makercolors = ['#e2f0d9','#fbe5d6','#dae3f3']
    linestyles = ['-', '-', '-']
    markers = ['o', '^', 's']

    figname = folder_path + '/strong_scalability.pdf'
    pdfpage, fig = pdf.plot_setup(figname, figsize=(8, 5), fontsize=22, font=('family', 'Tw Cen MT'))
    ax = fig.gca()
    for s, scheme in enumerate(names):
        ax.plot(
                nodes,
                algorithmic_scalability[scheme],
                marker=markers[s],
                markersize=14,
                markeredgecolor=colors[s],
                markerfacecolor=makercolors[s],
                markeredgewidth=3,
                color=colors[s],
                linestyle=linestyles[s],
                linewidth=3,
                label=schemes[s]
                )
        ax.set_xticks(nodes)
        ax.set_xlim(0, 270)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('Number of Nodes in 2D Torus Network')
        ax.set_ylabel('Normalized Runtime')
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


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('usage: ' + sys.argv[0] + ' folder_path')
        exit()
    main(sys.argv[1])
