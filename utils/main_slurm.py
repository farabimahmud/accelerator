import os
import shutil
import subprocess
from argparse import ArgumentParser

def get_slurm_arguments(args):
    slurm_arguments = ['sbatch']
    slurm_arguments.append('--nodes=1')
    slurm_arguments.append('--ntasks=1')
    slurm_arguments.append('--cpus-per-task=1')

    slurm_outpath = '{}/micro53_rebuttal/torus/{}_{}_error.log'.format(os.environ['SIMHOME'], args.run_name, args.scheme)
    slurm_arguments.append('--output={}'.format(slurm_outpath))

    python = shutil.which('python')

    program_arguments = []
    program_arguments.append(python)
    program_arguments.append('{}/src/simulate.py'.format(os.environ['SIMHOME']))
    program_arguments.append('--network={}'.format(args.network))
    program_arguments.append('--run-name={}'.format(args.run_name))
    program_arguments.append('--num-hmcs={}'.format(args.nodes))
    program_arguments.append('--booksim-config={}/src/booksim2/runfiles/{}'.format(os.environ['SIMHOME'], args.booksim_config))
    program_arguments.append('--allreduce={}'.format(args.allreduce))
    program_arguments.append('--outdir={}/micro53_rebuttal/torus'.format(os.environ['SIMHOME']))
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

    # copy and make the rundir
    #rundir = '{}/results/torus_logs/src'.format(os.environ['SIMHOME'])
    #shutil.copytree('{}/src'.format(os.environ['SIMHOME']), rundir)

    mlperfnns = ['AlphaGoZero', 'FasterRCNN', 'NCF_recommendation', 'Resnet50', 'Transformer']
    cnns = ['alexnet', 'Googlenet']
    mlperfpath = '{}/src/SCALE-Sim/topologies/mlperf'.format(os.environ['SIMHOME'])
    cnnpath = '{}/src/SCALE-Sim/topologies/conv_nets'.format(os.environ['SIMHOME'])
    nnpaths = []
    for nn in mlperfnns:
        nnpaths.append('{}/{}.csv'.format(mlperfpath, nn))
    for nn in cnns:
        nnpaths.append('{}/{}.csv'.format(cnnpath, nn))
    nns = mlperfnns + cnns
    minibatches = [256, 512, 1024, 2048, 4096, 8192]

    # run compute
    #for minibatch in minibatches:
    #    for i, nn in enumerate(nns):
    #        parser = ArgumentParser()
    #        args = parser.parse_args()

    #        args.nodes = 16
    #        args.network = nnpaths[i]
    #        args.minibatch = minibatch
    #        args.run_name = '{}_minibatch{}'.format(nn, minibatch)
    #        args.scheme = 'only_compute'
    #        args.only_compute = True
    #        args.only_allreduce = False
    #        args.booksim_config = 'torus4x4express.cfg'
    #        args.allreduce = 'ring' # useless
    #        args.kary = 2           # useless
    #        args.radix = 1          # useless
    #        args.message_size = 256

    #        command = ' '.join(get_slurm_arguments(args))

    #        print('Running slurm command: {}'.format(command))

    #        return_code = subprocess.run(command, env=os.environ, shell=True).returncode

    #        if return_code != 0:
    #            raise RuntimeError('{} -> {}'.format(command, return_code))

    # run allreduce
    allreduces = ['ring', 'dtree', 'mxnettree', 'mxnettree', 'mxnettree', 'multitree', 'multitree', 'multitree', 'ring']
    schemes = ['ring', 'dtree', 'mxnettree_alpha', 'mxnettree_beta', 'mxnettree_gamma', 'multitree_alpha', 'multitree_beta', 'multitree_gamma', 'ring_gamma']
    karies = [2, 2, 2, 2, 2, 5, 5, 5, 2]
    radices = [1, 1, 1, 4, 4, 1, 4, 4, 1]
    message_sizes = [256, 256, 256, 256, 0, 256, 256, 0, 0]
    booksim_configs_first = ['torus', 'torus', 'torus', 'ctorus', 'ctorus', 'torus', 'ctorus', 'ctorus', 'torus']
    booksim_configs_second = ['express.cfg', 'express.cfg', 'express.cfg', 'multitree.cfg', 'multitree.cfg', 'express.cfg', 'multitree.cfg', 'multitree.cfg', 'express.cfg']

    for n, nn in enumerate(nns):
        for i, allreduce in enumerate(allreduces):
            parser = ArgumentParser()
            args = parser.parse_args()

            args.nodes = 16
            args.network = nnpaths[n]
            args.allreduce = allreduce
            args.run_name = nn
            args.scheme = schemes[i]
            args.kary = karies[i]
            args.radix = radices[i]
            args.only_compute = False
            args.only_allreduce = True
            args.message_size = message_sizes[i]
            args.booksim_config = '{}4x4{}'.format(booksim_configs_first[i], booksim_configs_second[i])

            command = ' '.join(get_slurm_arguments(args))

            print('Running slurm command: {}'.format(command))

            return_code = subprocess.run(command, env=os.environ, shell=True).returncode

            if return_code != 0:
                raise RuntimeError('{} -> {}'.format(command, return_code))

if __name__ == '__main__':
    main()
