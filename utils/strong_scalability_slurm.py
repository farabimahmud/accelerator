import os
import shutil
import subprocess
from argparse import ArgumentParser

def get_slurm_arguments(args):
    slurm_arguments = ['sbatch']
    slurm_arguments.append('--nodes=1')
    slurm_arguments.append('--ntasks=1')
    slurm_arguments.append('--cpus-per-task=1')

    slurm_outpath = '{}/results/strong_scalability_logs/{}nodes_{}data_{}_error.log'.format(os.environ['SIMHOME'], args.nodes, args.data, args.scheme)
    slurm_arguments.append('--output={}'.format(slurm_outpath))

    python = shutil.which('python')

    program_arguments = []
    program_arguments.append(python)
    program_arguments.append('{}/src/simulate.py'.format(os.environ['SIMHOME']))
    program_arguments.append('--run-name={}nodes_{}data'.format(args.nodes,
args.data))
    program_arguments.append('--num-hmcs={}'.format(args.nodes))
    program_arguments.append('--booksim-config={}/src/booksim2/runfiles/{}'.format(os.environ['SIMHOME'], args.booksim_config))
    program_arguments.append('--allreduce={}'.format(args.allreduce))
    program_arguments.append('--outdir={}/results/strong_scalability_logs'.format(os.environ['SIMHOME']))
    program_arguments.append('--kary={}'.format(args.kary))
    program_arguments.append('--radix={}'.format(args.radix))
    program_arguments.append('--message-buffer-size=32')
    program_arguments.append('--message-size={}'.format(args.message_size))
    program_arguments.append('--sub-message-size=256')
    program_arguments.append('--only-allreduce')
    program_arguments.append('--synthetic-data-size={}'.format(args.data))

    slurm_arguments.append('--wrap=\"srun ' + ' '.join(program_arguments) + '\"')

    return slurm_arguments


def main():

    data = 8388608
    #allreduces = ['ring', 'mxnettree', 'multitree']
    #schemes = ['ring', 'mxnettree_beta', 'multitree_gamma']
    #karies = [2, 2, 5]
    #radices = [1, 4, 4]
    #message_sizes = [256, 256, 0]
    #booksim_configs_first = ['torus', 'ctorus', 'ctorus']
    #booksim_configs_second = ['express.cfg', 'multitree.cfg', 'multitree.cfg']
    allreduces = ['mxnettree']
    schemes = ['mxnettree_beta']
    karies = [2]
    radices = [4]
    message_sizes = [256]
    booksim_configs_first = ['ctorus']
    booksim_configs_second = ['multitree.cfg']

    #for dimension in range(4, 18, 2):
    for dimension in range(16, 18, 2):
        nodes = int(dimension * dimension)
        for i, allreduce in enumerate(allreduces):
            parser = ArgumentParser()
            #args = vars(parser.parse_args())
            args = parser.parse_args()

            args.nodes = nodes
            args.data = data
            args.allreduce = allreduce
            args.scheme = schemes[i]
            args.kary = karies[i]
            args.radix = radices[i]
            args.message_size = message_sizes[i]
            args.booksim_config = '{}{}x{}{}'.format(booksim_configs_first[i], dimension, dimension, booksim_configs_second[i])

            command = ' '.join(get_slurm_arguments(args))

            print('Running slurm command: {}'.format(command))

            #return_code = subprocess.run(command, env=os.environ, shell=True).returncode

            #if return_code != 0:
            #    raise RuntimeError('{} -> {}'.format(command, return_code))

if __name__ == '__main__':
    main()
