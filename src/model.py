class Model:
    def __init__(self, args):
        self.args = args
        self.num_layers = 0
        self.layers = []
        self.size = 0
        self.name = args.network.split('/')[-1].split('.')[0]

        self.parse_model()

    def parse_model(self):
        param_file = open(self.args.network, 'r')

        first = True

        print('\nModel loading ...')
        for row in param_file:
            if first:
                first = False
                continue

            elems = row.strip().split(',')

            # Do not continue if incomplete line
            if len(elems) < 9:
                if elems[0]:
                    print('Warn: incomplete model layer description: line ', self.num_layers + 1)
                    print(' -- ', row)
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
            layer_size = int(elems[3]) * int(elems[4]) * int(elems[6]) * int(elems[7])
            self.size += layer_size

        print('Model loading finished\n')
        #for l in range(self.num_layers):
        #    print('layer ', l, ': {name: ', self.layers[l]['name'],
        #            ', ifmap_h: ', self.layers[l]['ifmap_h'], ', ifmap_w: ', self.layers[l]['ifmap_w'],
        #            ', filter_h: ', self.layers[l]['filter_h'], ', filter_w', self.layers[l]['filter_w'],
        #            ', num_channels: ', self.layers[l]['num_channels'],
        #            ', num_filters: ', self.layers[l]['num_filters'],
        #            ', stride: ', self.layers[l]['stride'], '}')

    # parse_model() end
