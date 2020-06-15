import os
import shutil
import subprocess
from argparse import ArgumentParser

def get_slurm_arguments(args):
    slurm_arguments = ['sbatch']
    slurm_arguments.append('--nodes=1')
    slurm_arguments.append('--ntasks=1')
    slurm_arguments.append('--cpus-per-task=1')

    slurm_outpath = '{}/micro53_rebuttal/bigraph/{}_{}_error.log'.format(os.environ['SIMHOME'], args.run_name, args.scheme)
    slurm_arguments.append('--output={}'.format(slurm_outpath))

    python = shutil.which('python')

    program_arguments = []
    program_arguments.append(python)
    program_arguments.append('{}/src/simulate.py'.format(os.environ['SIMHOME']))
    program_arguments.append('--network={}'.format(args.network))
    program_arguments.append('--run-name={}'.format(args.run_name))
    program_arguments.append('--num-hmcs={}'.format(args.nodes))
    program_arguments.append('--booksim-network=bigraph')
    program_arguments.append('--booksim-config={}/src/booksim2/runfiles/{}'.format(os.environ['SIMHOME'], args.booksim_config))
    program_arguments.append('--bigraph-m=4')
    program_arguments.append('--bigraph-n=8')
    program_arguments.append('--allreduce={}'.format(args.allreduce))
    program_arguments.append('--outdir={}/micro53_rebuttal/bigraph'.format(os.environ['SIMHOME']))
    program_arguments.append('--kary={}'.format(args.kary))
    program_arguments.append('--radix={}'.format(args.radix))
    program_arguments.append('--message-buffer-size=32')
    program_arguments.append('--message-size={}'.format(args.message_size))
    program_arguments.append('--sub-message-size=256')
    if args.only_compute == True:
        program_arguments.append('--mini-batch-size={}'.format(args.minibatch))
        program_arguments.append('--only-compute')
    if args.only_allreduce == True:
        program_arguments.append('--only-allreduce')

    slurm_arguments.append('--wrap=\"srun ' + ' '.join(program_arguments) + '\"')

    return slurm_arguments


def main():

    mlperfnns = ['AlphaGoZero', 'FasterRCNN', 'NCF_recommendation', 'Resnet50', 'Transformer', 'Transformer_short']
    cnns = ['alexnet', 'Googlenet']
    mlperfpath = '{}/src/SCALE-Sim/topologies/mlperf'.format(os.environ['SIMHOME'])
    cnnpath = '{}/src/SCALE-Sim/topologies/conv_nets'.format(os.environ['SIMHOME'])
    nnpaths = []
    for nn in mlperfnns:
        nnpaths.append('{}/{}.csv'.format(mlperfpath, nn))
    for nn in cnns:
        nnpaths.append('{}/{}.csv'.format(cnnpath, nn))
    nns = mlperfnns + cnns
    #minibatches = [256, 512, 1024, 2048, 4096, 8192]

    # run allreduce
    allreduces = ['multitree', 'multitree', 'hdrm', 'hdrm']
    schemes = ['multitree', 'multitree_gamma', 'hdrm', 'hdrm_gamma']
    karies = [2, 2, 2, 2]
    radices = [1, 1, 1, 1]
    only_allreduce= [False, True, True, True]
    message_sizes = [256, 0, 256, 0]
    booksim_configs = ['bigraph.cfg', 'bigraph.cfg', 'bigraph.cfg', 'bigraph.cfg']

    for n, nn in enumerate(nns):
        for i, allreduce in enumerate(allreduces):
            parser = ArgumentParser()
            args = parser.parse_args()

            args.nodes = 32
            args.network = nnpaths[n]
            args.allreduce = allreduce
            args.run_name = nn
            args.scheme = schemes[i]
            args.kary = karies[i]
            args.radix = radices[i]
            args.only_compute = False
            args.only_allreduce = only_allreduce[i]
            args.message_size = message_sizes[i]
            args.booksim_config = booksim_configs[i]

            command = ' '.join(get_slurm_arguments(args))

            print('Running slurm command: {}'.format(command))

            return_code = subprocess.run(command, env=os.environ, shell=True).returncode

            if return_code != 0:
                raise RuntimeError('{} -> {}'.format(command, return_code))

if __name__ == '__main__':
    main()
