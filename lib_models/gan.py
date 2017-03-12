# Modified from htps://github.com/ppaquette/ift-6266-project/blob/master/models/gan_v2.py
import collections
import numpy as np
import theano, theano.tensor as T
from lib.initializations import *
from lib.ops import *
from lib.theano_utils import *
from lib.updates import *
from lib.data_utils import *

class GenerativeAdversarialNetwork(object):
    """ Generative Adversarial Network Implementation """
    def build_model(self):

        print("Building model...")

        self.hparams = self.get_hparams()
        hp = self.hparams
        #locals().update(hp)


        adam_b1 = 0.5
        l2_reg = 1e-5
        batch_size = hp['batch_size']
        nb_batch_store_gpu = hp['nb_batch_store_gpu']
        nb_batch_train = 82611 // batch_size
        nb_batch_valid = 30000 // batch_size

        hist_alpha = 1. / max(100., 2. * nb_batch_train)


        Z_dim = hp['Z_dim']
        Z = T.matrix('Z', dtype='float32')
        Y = T.tensor4(name='Y', dtype='float32')
        masked_img = T.tensor4(name='mask_img', dtype='float32')
        index = T.iscalar('index')



        # space for storing dataset on GPU
        self.gpu_dataset_X = shared0s(shape=(batch_size * nb_batch_store_gpu, 3, 64, 64))
        self.gpu_dataset_Y = shared0s(shape=(batch_size * nb_batch_store_gpu, 3, 64, 64))

        # Define parameters
        self.params = collections.OrderedDict()

        # learning rate
        self.params['hyper_learning_rate'] = sharedX(hp['learning_rate'], name='learning_rate', dtype='float32')

        # ---------
        # Generator
        # ---------

        # Convolutions
        self.params['gen_00_W'] = he(hp['gen_00_f'], name='gen_00_W', fan_in=np.prod(hp['gen_00_f'][1:]))
        self.params['gen_00_b'] = constant((hp['gen_00_f'][0],), name='gen_00_b')

        self.params['gen_01_W'] = he(hp['gen_01_f'], name='gen_01_W', fan_in=np.prod(hp['gen_01_f'][1:]))
        self.params['gen_01_b'] = constant((hp['gen_01_f'][0],), name='gen_01_b')

        self.params['gen_02_W'] = he(hp['gen_02_f'], name='gen_02_W', fan_in=np.prod(hp['gen_02_f'][1:]))
        self.params['gen_02_b'] = constant((hp['gen_02_f'][0],), name='gen_02_b')

        # Deconvolutions
        self.params['gen_03_W'] = he(hp['gen_03_f'], name='gen_03_W', fan_in=np.prod(hp['gen_03_f'][1:]))
        self.params['gen_03_b'] = constant((hp['gen_03_f'][1],), name='gen_03_b')

        self.params['gen_04_W'] = he(hp['gen_04_f'], name='gen_04_W', fan_in=np.prod(hp['gen_04_f'][1:]))
        self.params['gen_04_b'] = constant((hp['gen_04_f'][1],), name='gen_04_b')

        self.params['gen_05_W'] = he(hp['gen_05_f'], name='gen_05_W', fan_in=np.prod(hp['gen_05_f'][1:]))
        self.params['gen_05_b'] = constant((hp['gen_05_f'][1],), name='gen_05_b')

        self.params['gen_06_W'] = he(hp['gen_06_f'], name='gen_06_W', fan_in=np.prod(hp['gen_06_f'][1:]))
        self.params['gen_06_b'] = constant((hp['gen_06_f'][1],), name='gen_06_b')

        # -------------
        # Discriminator
        # -------------

        self.params['dis_00_W'] = he(hp['dis_00_f'], name='dis_00_W', fan_in=np.prod(hp['dis_00_f'][1:]))
        self.params['dis_00_b'] = constant((hp['dis_00_f'][0],), name='dis_00_b')

        self.params['dis_01_W'] = he(hp['dis_01_f'], name='dis_01_W', fan_in=np.prod(hp['dis_01_f'][1:]))
        self.params['dis_01_b'] = constant((hp['dis_01_f'][0],), name='dis_01_b')

        self.params['dis_02_W'] = he(hp['dis_02_f'], name='dis_02_W', fan_in=np.prod(hp['dis_02_f'][1:]))
        self.params['dis_02_b'] = constant((hp['dis_02_f'][0],), name='dis_02_b')

        # minibatch discrimination
        self.params['dis_03_W'] = he(hp['dis_03_f'], name='dis_03_W', fan_in=np.prod(hp['dis_03_f'][1:]))
        self.params['dis_03_b'] = constant((hp['dis_03_f'][1],), name='dis_03_b')

        self.params['dis_04_W'] = he(hp['dis_04_f'], name='dis_04_W', fan_in=np.prod(hp['dis_04_f'][1:]))
        self.params['dis_04_b'] = constant((hp['dis_04_f'][1],), name='dis_04_b')

        # -----------------
        # Historical Avg
        # -----------------

        # Generator - Historical Averaging
        list_gen_params = [param for param in self.params.keys() if param.startswith('gen_')]
        g_avg_updates =[]
        for gen_param in list_gen_params:
            self.params['avg_'+gen_param] = shared0s(self.params[gen_param].get_value().shape)
            g_avg_updates.append((self.params['avg_'+gen_param],
                                  hist_alpha * self.params[gen_param] +
                                  (1. - hist_alpha) * self.params['avg_'+gen_param]))

        # Discriminator - Historical Averaging
        list_disc_params = [param for param in self.params.keys() if param.startswith('dis_')]
        d_avg_updates =[]
        for disc_param in list_disc_params:
            self.params['avg_'+disc_param] = shared0s(self.params[disc_param].get_value().shape)
            d_avg_updates.append((self.params['avg_'+disc_param],
                                  hist_alpha * self.params[disc_param] +
                                  (1. - hist_alpha) * self.params['avg_'+disc_param]))


        p = self.params

        # Generator
        def generator(Z, masked_img):
            """ Generator model """

            # encode the contour of the image
            input = masked_img

            gen_00 = relu(batchnorm(conv(input, hp['gen_00_f'], p['gen_00_W']))
                         + p['gen_00_b'].dimshuffle('x', 0, 'x', 'x'))
            gen_00 = pool2D(gen_00)                                                     # (b, 64, 30, 30)

            gen_01 = relu(batchnorm(conv(gen_00, hp['gen_01_f'], p['gen_01_W']))
                         + p['gen_01_b'].dimshuffle('x', 0, 'x', 'x'))
            gen_01 = pool2D(gen_01)                                                     # (b, 64, 13, 13)

            gen_02 = relu(batchnorm(conv(gen_01, hp['gen_02_f'], p['gen_02_W']))
                         + p['gen_02_b'].dimshuffle('x', 0, 'x', 'x'))
            gen_02 = pool2D(gen_02)                                                     # (b, 64, 4, 4)

            # concatenate the encoded information from mask_img with the noise Z
            prior = concat([Z, gen_02.flatten(2)]).reshape(shape=(batch_size, 281, 2, 2))

            # Generate X
            gen_03 = relu(batchnorm(deconv(prior, hp['gen_03_f'], hp['gen_03_i'],
                    p['gen_03_W'])) + p['gen_03_b'].dimshuffle('x', 0, 'x', 'x'))     # f = (2, 2)
            gen_03 = Unpool2D(gen_03)                                                   # (b, 64, 6, 6)

            gen_04 = relu(batchnorm(deconv(gen_03, hp['gen_04_f'], hp['gen_04_i'],
                    p['gen_04_W'])) + p['gen_04_b'].dimshuffle('x', 0, 'x', 'x'))       # f = (2, 2)
            gen_04 = Unpool2D(gen_04)                                                   # (b, 64, 14, 14)

            gen_05 = relu(batchnorm(deconv(gen_04, hp['gen_05_f'], hp['gen_05_i'],
                    p['gen_05_W'])) + p['gen_05_b'].dimshuffle('x', 0, 'x', 'x'))       # f = (2, 2)
            gen_05 = Unpool2D(gen_05)                                                   # (b, 64, 30, 30)

            gen_06 = relu(batchnorm(deconv(gen_05, hp['gen_06_f'], hp['gen_06_i'],
                    p['gen_06_W'])) + p['gen_06_b'].dimshuffle('x', 0, 'x', 'x'))       # f = (3, 3)
            gen_06 = Unpool2D(gen_06)                                                   # (b, 3, 64, 64)

            return gen_06


        # Discriminator
        # Adapted from https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py#L132
        def minibatch(d, W, b):
            """ Minibatch Discrimination """
            mb_h0 = T.dot(d.flatten(2), W)                                                     # b, nb_kernels * kernel_dim
            mb_h0 = mb_h0.reshape((d.shape[0], hp['mb_nb_kernels'], hp['mb_kernel_dim']))      # b, nb_kernel, kernel_dim
            mb_h1 = mb_h0.dimshuffle(0, 1, 2, 'x') - mb_h0.dimshuffle('x', 1, 2, 0)
            mb_h1 = T.sum(abs(mb_h1), axis=2) + 1e6 * T.eye(d.shape[0]).dimshuffle(0,'x',1)
            mb_h2 = T.sum(T.exp(-mb_h1), axis=2) + b                                              # b, nb_kernel
            mb_h2 = mb_h2.dimshuffle(0, 1, 'x', 'x')
            mb_h2 = T.repeat(mb_h2, 4, axis=2)
            mb_h2 = T.repeat(mb_h2, 4, axis=3)
            return mb_h2

        def discriminator(X):
            """ Discriminator Model """

            # Combine the real "contour" with the generated images
            #mask = T.zeros((batch_size, 3, 64, 64), dtype='float32')
            #mask = T.set_subtensor(mask[:, :, 16:48, 16:48], 1.)
            #images = mask * X + (1 - mask) * Y
            # ---------------------------------------------------

            images = X

            dis_00 = lrelu(batchnorm(conv(images, hp['dis_00_f'], p['dis_00_W']))
                           + p['dis_00_b'].dimshuffle('x', 0, 'x', 'x'))
            dis_00 = pool2D(dis_00)                                                             # (b, 256, 30, 30)

            dis_01 = lrelu(batchnorm(conv(dis_00, hp['dis_01_f'], p['dis_01_W']))
                           + p['dis_01_b'].dimshuffle('x', 0, 'x', 'x'))
            dis_01 = pool2D(dis_01)                                                             # (b, 512, 13, 13)

            dis_02 = lrelu(batchnorm(conv(dis_01, hp['dis_02_f'], p['dis_02_W']))
                           + p['dis_02_b'].dimshuffle('x', 0, 'x', 'x'))
            dis_02 = pool2D(dis_02)                                                             # (b, 1024, 4, 4)

            dis_mb = minibatch(dis_02, W=p['dis_03_W'], b=p['dis_03_b'])
            dis_mb = T.flatten(dis_mb, outdim=2)

            dis_Y = T.nnet.sigmoid(T.dot(dis_mb, p['dis_04_W']) + p['dis_04_b'])
            return dis_Y

        X = generator(Z, masked_img)
        p_data = discriminator(Y)
        p_gen = discriminator(X)

        # Historical Averaging
        g_avg_cost, d_avg_cost = 0., 0.
        nb_g_avg_param, nb_d_avg_param = 0., 0.
        for param in self.params.keys():
            if param.startswith('gen_'):
                g_avg_cost += MeanSquaredError(self.params[param], self.params['avg_'+param])
                nb_g_avg_param += 1
            if param.startswith('disc_'):
                d_avg_cost += MeanSquaredError(self.params[param], self.params['avg_'+param])
                nb_d_avg_param += 1
        g_avg_cost = g_avg_cost / max(1., nb_g_avg_param)
        d_avg_cost = d_avg_cost / max(1., nb_d_avg_param)

        mask = T.zeros((batch_size, 3, 64, 64), dtype='float32')
        mask = T.set_subtensor(mask[:, :, 16:48, 16:48], 1.)

        # Generator Costs & Updates
        g_cost_dis = BinaryCrossEntropy(p_gen, 0.9)
        g_border_cost = MeanSquaredError(X * (1. - mask), Y * (1. - mask))
        g_total_cost = g_cost_dis * 0.1
        g_total_cost += g_border_cost * 1.0
        g_total_cost += g_avg_cost * 1.0

        g_params = [self.params[k] for k in self.params.keys() if k.startswith('gen_')]

        g_updater = Adam(lr=self.params['hyper_learning_rate'],
                         b1=adam_b1, clipnorm=10.,
                         regularizer=Regularizer(l2=l2_reg))

        g_updates = g_updater(g_params, g_total_cost) + g_avg_updates
        g_border_updates = g_updater(g_params, g_border_cost)


        # Discriminator Costs & Updates
        d_cost_data = BinaryCrossEntropy(p_data, 0.9)
        d_cost_gen = BinaryCrossEntropy(p_gen, 0.0)
        d_total_cost = d_cost_data * 1.0
        d_total_cost += d_cost_gen * 0.5
        d_total_cost += d_avg_cost * 1.0

        d_params = [self.params[k] for k in self.params.keys()
                    if k.startswith('dis_')]
        d_updater = Adam(lr=self.params['hyper_learning_rate'],
                         b1=adam_b1, clipnorm=10.,
                         regularizer=Regularizer(l2=l2_reg))
        d_updates = d_updater(d_params, d_total_cost) + d_avg_updates

        d_g_updates = d_updates + g_updates


        # -------------------
        # Compiling Functions
        # -------------------

        print('Compiling train_gen_fn()...')
        self.train_gen_fn = theano.function(
            inputs=[index, Z],
            outputs=[g_total_cost, g_cost_dis, g_border_cost, g_avg_cost],
            updates=g_updates,
            givens={
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size],
                masked_img: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size]
            }
        )
        print('...Done Compiling train_gen_fn()')

        print('Compiling train_gen_border_fn()...')
        self.train_gen_border_fn = theano.function(
            inputs=[index, Z],
            outputs=[g_border_cost],
            updates=g_border_updates,
            givens={
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size],
                masked_img: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size]
            }
        )
        print('...Done Compiling train_gen_border_fn()')

        print('Compiling train_dis_fn()...')
        self.train_dis_fn = theano.function(
            inputs=[index, Z],
            outputs=[d_total_cost, d_cost_data, d_cost_gen, d_avg_cost],
            updates=d_updates,
            givens={
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size],
                masked_img: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size]
            }
        )
        print('...Done Compiling train_dis_fn()')

        print('Compiling train_d_g_fn()...')
        self.train_g_d_fn = theano.function(
            inputs=[index, Z],
            outputs=[g_total_cost, d_total_cost, g_cost_dis, g_border_cost,
                     g_avg_cost, d_cost_data, d_cost_gen, d_avg_cost],
            updates=d_g_updates,
            givens={
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size],
                masked_img: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size]
            }
        )
        print('...Done Compiling train_d_g_fn()')

    def train_model(self):
        """ Train the GAN """
        print('Training Model...')
        rng = np.random.RandomState(1)
        hp = self.hparams
        batch_size = hp['batch_size']
        nb_batch_store_gpu = hp['nb_batch_store_gpu']
        nb_batch_train = 82611 // batch_size
        nb_batch_valid = 30000 // batch_size
        nb_epochs = hp['nb_epochs']
        z_dim = hp['z_dim']

        learning_rate_epoch = hp['learning_rate_epoch']
        learning_rate_adj = hp['learning_rate_adj']
        initial_learning_rate = hp['learning_rate']
        last_cost_g = []
        last_cost_d = []

        patience = hp['patience']
        patience_increase = 2
        improvement_threshold = 0.995

        DONE_LOOPING = False
        epoch = 0
        self.params['hyper_learning_rate'].set_value(initial_learning_rate)
        batch_idx = 0
        while epoch < nb_epochs and not DONE_LOOPING:
            epoch += 1
            masterbatch_idx = 0

            # loop through the GPU batches
            for gpu_batch in iterate_minibatches(batch_size * nb_batch_store_gpu, 'train2014'):

                self.gpu_dataset_X.set_value(gpu_batch[0])
                self.gpu_dataset_Y.set_value(gpu_batch[1])


                for idx in range(nb_batch_store_gpu):
                    batch_idx += 1
                    Z = rng.normal(0, 1, size=(batch_size, z_dim)).astype('float32')

                    cost_g, cost_d, g_cost_d, g_border_cost, g_avg_cost, \
                        d_cost_data, d_cost_gen, d_avg_cost = self.train_g_d_fn(idx, Z)
                    last_cost_g.append(cost_g)
                    last_cost_g = last_cost_g[-1000:]
                    last_cost_d.append(cost_d)
                    last_cost_d = last_cost_d[-1000:]
                    print('Batch %04d/%04d - Epoch %04d -- Loss G/D: %.4f/%.4f - Min/Max (1000) %.4f/%.4f - %.4f/%.4f' %
                          (batch_idx, nb_batch_train, epoch, cost_g, cost_d, min(last_cost_g),
                           max(last_cost_g), min(last_cost_d), max(last_cost_d)))

































    def get_hparams(self):
        return {
            "batch_size": 128,
            "nb_batch_store_gpu":   5,  # # of batch to store on the GPU
            "k" : 1,            # # of discrim updates for each gen update
            "l2" : 2.5e-5,      # l2 weight decay
            "b1" : 0.5,         # momentum term of adam
            "learning_rate" : 0.0002,      # initial learning rate for adam
            'learning_rate_epoch': 10,
            'learning_rate_adj': 0.8,
            'mb_kernel_dim': 5,
            'mb_nb_kernels': 100,

            # hyper-parameters for the model

            # ---------
            # Generator
            # ---------

            # Convolutions
            'gen_00_f': (64, 3, 5, 5),
            'gen_01_f': (64, 64, 5, 5),
            'gen_02_f': (64, 64, 5, 5),

            # Deconvolutions
            'gen_03_f': (281, 64, 2, 2),
            'gen_03_i': (128, 64, 6, 6),

            'gen_04_f': (64, 64, 2, 2),
            'gen_04_i': (128, 64, 14, 14),

            'gen_05_f': (64, 64, 2, 2),
            'gen_05_i': (128, 64, 30, 30),

            'gen_06_f': (64, 3, 3, 3),
            'gen_06_i': (128, 3, 64, 64),

            # -------------
            # Discriminator
            # -------------

            # Convolutions
            'dis_00_f': (256, 3, 5, 5),
            'dis_01_f': (512, 256, 5, 5),
            'dis_02_f': (1024, 512, 5, 5),

            # Minibatch Discrimination
        }

if __name__ == '__main__':
    network = GenerativeAdversarialNetwork()
    network.build_model()
    network.train_model()
