import logging
import numpy as np

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, args):
        self.args = args
        if args.synthetic_data_size:
            self.size = args.synthetic_data_size
            assert args.only_allreduce
        else:
            self.size = 0
            self.num_layers = 0
            self.layers = []
            self.layer_size = []
            self.name = args.network.split('/')[-1].split('.')[0]

            self.inference_cycles = None
            self.backprop_cycles = None
            self.inference_layer_wise_cycles = None
            self.backprop_layer_wise_cycles = None

            self.parse_model()

    def parse_model(self):
        param_file = open(self.args.network, 'r')

        first = True

        logger.info('\nModel loading ...')
        for row in param_file:
            if first:
                first = False
                continue

            elems = row.strip().split(',')

            # Do not continue if incomplete line
            if len(elems) < 9:
                if elems[0]:
                    logger.warn('Warn: incomplete model layer description: line {}'.format(self.num_layers + 1))
                    logger.warn(' -- {}'.format(row))
                continue


            self.layers.append({})
            self.layers[self.num_layers]['name'] = elems[0]
            self.layers[self.num_layers]['ifmap_h'] = int(elems[1])
            self.layers[self.num_layers]['ifmap_w'] = int(elems[2])
            self.layers[self.num_layers]['filter_h'] = int(elems[3])
            self.layers[self.num_layers]['filter_w'] = int(elems[4])
            self.layers[self.num_layers]['num_channels'] = int(elems[5])
            self.layers[self.num_layers]['num_filters'] = int(elems[6])
            self.layers[self.num_layers]['stride'] = int(elems[7])

            self.num_layers += 1
            layer_size = int(elems[3]) * int(elems[4]) * int(elems[5]) * int(elems[6])
            self.size += layer_size
            self.layer_size.append(layer_size)

        self.inference_layer_wise_cycles = np.zeros(self.num_layers)
        self.backprop_layer_wise_cycles = np.zeros(self.num_layers)

        logger.info('Model loading finished\n')
        for l in range(self.num_layers):
            logger.debug('layer: {}: [name: {}, ifmap_h: {}, ifmap_w: {},'
                         'filter_h: {}, filter_w: {}, num_channels: {},'
                         'num_filters: {}, stride: {}]'.format(l,
                             self.layers[l]['name'], self.layers[l]['ifmap_h'],
                             self.layers[l]['ifmap_w'],
                             self.layers[l]['filter_h'],
                             self.layers[l]['filter_w'],
                             self.layers[l]['num_channels'],
                             self.layers[l]['num_filters'],
                             self.layers[l]['stride']))
    # parse_model() end
