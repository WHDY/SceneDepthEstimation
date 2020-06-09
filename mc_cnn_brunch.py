import tensorflow as tf
import numpy as np

class Net:
    def __init__(self, inputs,
                 weights_path='DEFAULT',
                 input_patch_size=11,
                 num_of_conv_layers=4,
                 num_of_conv_feature_maps=64,
                 conv_kernel_size=3,
                 batch_size=128,
                 is_branch=False):
        '''   initialize the parameters   '''
        self.input = inputs
        self.input_patch_size = input_patch_size
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_conv_feature_maps = num_of_conv_feature_maps
        self.conv_kernel_size = conv_kernel_size
        self.batch_size = batch_size
        self.is_branch = is_branch

        if weights_path == 'DEFAULT':
            self.weights_path = 'pretrain.npy'
        else:
            self.weights_path = weights_path

        # call the construction function to build the network
        self.construct()


    def construct(self):
        channels = 1
        bs = self.batch_size
        ks = self.conv_kernel_size
        nl = self.num_of_conv_layers
        nf = self.num_of_conv_feature_maps

        self.conv1 = conv(self.input, ks, ks, channels, nf, 1, 1, padding='VALID',\
                              activation_function='RELU', name='conv1', already_have=self.is_branch)

        for _ in range(2, nl):
            setattr(self, 'conv{}'.format(_), conv(getattr(self, 'conv{}'.format(_-1)), ks, ks, nf, nf, 1, 1, \
                                                   padding='VALID', activation_function='RELU', name='conv{}'.format(_), already_have=self.is_branch))

        setattr(self, 'conv{}'.format(nl), conv(getattr(self, 'conv{}'.format(nl-1)), ks, ks, nf, nf, 1, 1,\
                                                 padding='VALID', activation_function='NONE', name='conv{}'.format(nl), already_have = self.is_branch))

        self.features = tf.nn.l2_normalize(getattr(self, 'conv{}'.format(nl)), dim=-1, name='normalize')


    def load_initial_weights(self, session):
        all_vars = tf.trainable_variables()
        weights_dict = np.load(self.weights_path, encoding='bytes', allow_pickle=True).item()

        for name in weights_dict:
            print('restoring var{}...'.format(name))
            var = [var for var in all_vars if var.name == name][0]
            session.run(var.assign(weights_dict[name]))


    def save_weights_dict(self, session, file_name='pretrain.npy'):
        save_vars = tf.trainable_variables()
        weights_dict = {}
        for var in save_vars:
            weights_dict[var.name] = session.run(var)
        np.save(file_name, weights_dict)
        print('weights saved in file {}'.format(file_name))


def conv(input, filter_height, filter_width, input_channels, num_filters, \
         stride_y, stride_x, name, padding='SAME', activation_function='RELU', groups=1, already_have = False):
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        if already_have == True:
            scope.reuse_variables()
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters], trainable=True)
        biases = tf.get_variable('biases', shape=[num_filters], trainable=True)
        if groups == 1:
            conv = convolve(input, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=input)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)

        wx_plus_biases = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

        if activation_function == 'RELU':
            output = tf.nn.relu(wx_plus_biases, name=scope.name)
        elif activation_function == 'NONE':
            output = tf.identity(wx_plus_biases, name=scope.name)
    return output


def fc(input, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        wx_plus_biases = tf.nn.xw_plus_b(input, weights, biases, name=scope)

    if relu==True:
        output=tf.nn.relu(wx_plus_biases)
        return output
    else:
        return wx_plus_biases


if __name__=='__main__':
    x = tf.placeholder(tf.float32, [128, 11, 11, 1])
    y = tf.placeholder(tf.float32, [128, 11, 11, 1])
    z = tf.placeholder(tf.float32, [128, 11, 11, 1])
    brunchl = Net(x, input_patch_size=11, is_branch=False)
    brunchrp = Net(y, input_patch_size=11, is_branch=True)
    brunchrn = Net(z, input_patch_size=11, is_branch=True)
    var_list = tf.trainable_variables()
    for var in var_list:
        print('{}'.format(var.name))


