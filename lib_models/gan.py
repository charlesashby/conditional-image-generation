# Modified from htps://github.com/ppaquette/ift-6266-project/blob/master/models/gan_v2.py
import collections
import numpy as np
import theano, theano.tensor as T
from lib.initializations import *
from lib.ops import *
from lib.theano_utils import *
from lib.updates import *
from lib.data_utils import *
import cPickle
from theano.ifelse import ifelse
import gensim

theano.config.exception_verbosity = 'high'


class GenerativeAdversarialNetwork(object):
    """ Generative Adversarial Network Implementation """

    def build_model(self):

        print("Building model...")

        self.hparams = self.get_hparams()
        hp = self.hparams
        # locals().update(hp)


        adam_b1 = 0.5
        l2_reg = 1e-5
        batch_size = hp['batch_size']
        nb_batch_store_gpu = hp['nb_batch_store_gpu']
        nb_batch_train = 82611 // batch_size
        nb_batch_valid = 30000 // batch_size

        hist_alpha = 1. / max(100., 2. * nb_batch_train)

        Z = T.matrix('Z', dtype='float32')
        Y = T.tensor4(name='Y', dtype='float32')
        embedding = T.matrix(name='embed', dtype='float32')

        masked_img = T.tensor4(name='mask_img', dtype='float32')
        index = T.iscalar('index')
        in_training = T.bscalar('in_training')

        mask = T.zeros((batch_size, 3, 64, 64), dtype='float32')
        mask = T.set_subtensor(mask[:, :, 16:48, 16:48], 1.)

        # space for storing dataset on GPU
        self.gpu_dataset_X = shared0s(shape=(batch_size * nb_batch_store_gpu, 3, 64, 64))
        self.gpu_dataset_Y = shared0s(shape=(batch_size * nb_batch_store_gpu, 3, 64, 64))
        self.gpu_dataset_emb = shared0s(shape=(batch_size * nb_batch_store_gpu, 4800))

        # Define parameters
        self.params = collections.OrderedDict()

        # learning rate
        #self.params['hyper_learning_rate'] = sharedX(hp['learning_rate'], name='learning_rate', dtype='float32')

        # -----------
        # SqueezeNet
        # -----------

        # SqueezeNet Parameters
        # Reference: SqueezeNet: AlexNet-Level Accuracy with 50x Fewer Parameters and <1MB Model Size, Iandola et al. 2016
        #            Exploration of the Effect of Residual Connection on top of SqueezeNet. Shen and Han.
        s_03, e_03 = 16, 64
        s_04, e_04 = 16, 64
        s_05, e_05 = 32, 128
        s_07, e_07 = 32, 128
        s_08, e_08 = 48, 192
        s_09, e_09 = 48, 192
        s_10, e_10 = 64, 256
        s_12, e_12 = 64, 256
        self.params['squ_01_conv'] = He(shape=(96, 3, 3, 3), fan_in=(96 * 5 * 5), name='squ_01_conv')  # b, 96, 64, 64
        self.params['squ_01_g'] = normal(shape=(96,), mean=1.0, std_dev=0.02, name='squ_01_g')
        self.params['squ_01_b'] = constant(shape=(96,), c=0., name='squ_01_b')
        self.params['squ_03_w11s'] = He(shape=(s_03, 96, 1, 1), fan_in=(s_03 * 1 * 1),
                                        name='squ_03_w11s')  # b, s_03, 32, 32
        self.params['squ_03_w11e'] = He(shape=(e_03, s_03, 1, 1), fan_in=(e_03 * 1 * 1),
                                        name='squ_03_w11e')  # b, e_03, 32, 32
        self.params['squ_03_w33e'] = He(shape=(e_03, s_03, 3, 3), fan_in=(e_03 * 3 * 3),
                                        name='squ_03_w33e')  # b, e_03, 32, 32
        self.params['squ_03_w11b'] = He(shape=(2 * e_03, 96, 1, 1), fan_in=(2 * e_03 * 1 * 1),
                                        name='squ_03_w11b')  # b, 2 * e_03, 32, 32
        self.params['squ_03_g'] = normal(shape=(2 * e_03,), mean=1.0, std_dev=0.02, name='squ_03_g')
        self.params['squ_03_b'] = constant(shape=(2 * e_03,), c=0., name='squ_03_b')
        self.params['squ_04_w11s'] = He(shape=(s_04, 2 * e_03, 1, 1), fan_in=(s_04 * 1 * 1),
                                        name='squ_04_w11s')  # b, s_03, 32, 32
        self.params['squ_04_w11e'] = He(shape=(e_04, s_04, 1, 1), fan_in=(e_04 * 1 * 1), name='squ_04_w11e')
        self.params['squ_04_w33e'] = He(shape=(e_04, s_04, 3, 3), fan_in=(e_04 * 3 * 3), name='squ_03_w33e')
        self.params['squ_04_g'] = normal(shape=(2 * e_04,), mean=1.0, std_dev=0.02, name='squ_04_g')
        self.params['squ_04_b'] = constant(shape=(2 * e_04,), c=0., name='squ_04_b')
        self.params['squ_05_w11s'] = He(shape=(s_05, 2 * e_04, 1, 1), fan_in=(s_05 * 1 * 1), name='squ_05_w11s')
        self.params['squ_05_w11e'] = He(shape=(e_05, s_05, 1, 1), fan_in=(e_05 * 1 * 1), name='squ_05_w11e')
        self.params['squ_05_w33e'] = He(shape=(e_05, s_05, 3, 3), fan_in=(e_05 * 3 * 3), name='squ_05_w33e')
        self.params['squ_05_w11b'] = He(shape=(2 * e_05, 2 * e_04, 1, 1), fan_in=(2 * e_05 * 1 * 1), name='squ_05_w11b')
        self.params['squ_05_g'] = normal(shape=(2 * e_05,), mean=1.0, std_dev=0.02, name='squ_05_g')
        self.params['squ_05_b'] = constant(shape=(2 * e_05,), c=0., name='squ_05_b')
        self.params['squ_07_w11s'] = He(shape=(s_07, 2 * e_05, 1, 1), fan_in=(s_07 * 1 * 1), name='squ_07_w11s')
        self.params['squ_07_w11e'] = He(shape=(e_07, s_07, 1, 1), fan_in=(e_07 * 1 * 1), name='squ_07_w11e')
        self.params['squ_07_w33e'] = He(shape=(e_07, s_07, 3, 3), fan_in=(e_07 * 3 * 3), name='squ_07_w33e')
        self.params['squ_07_g'] = normal(shape=(2 * e_07,), mean=1.0, std_dev=0.02, name='squ_07_g')
        self.params['squ_07_b'] = constant(shape=(2 * e_07,), c=0., name='squ_07_b')
        self.params['squ_08_w11s'] = He(shape=(s_08, 2 * e_07, 1, 1), fan_in=(s_08 * 1 * 1), name='squ_08_w11s')
        self.params['squ_08_w11e'] = He(shape=(e_08, s_08, 1, 1), fan_in=(e_08 * 1 * 1), name='squ_08_w11e')
        self.params['squ_08_w33e'] = He(shape=(e_08, s_08, 3, 3), fan_in=(e_08 * 3 * 3), name='squ_03_w33e')
        self.params['squ_08_w11b'] = He(shape=(2 * e_08, 2 * e_07, 1, 1), fan_in=(2 * e_08 * 1 * 1), name='squ_08_w11b')
        self.params['squ_08_g'] = normal(shape=(2 * e_08,), mean=1.0, std_dev=0.02, name='squ_08_g')
        self.params['squ_08_b'] = constant(shape=(2 * e_08,), c=0., name='squ_08_b')
        self.params['squ_09_w11s'] = He(shape=(s_09, 2 * e_08, 1, 1), fan_in=(s_09 * 1 * 1), name='squ_09_w11s')
        self.params['squ_09_w11e'] = He(shape=(e_09, s_09, 1, 1), fan_in=(e_09 * 1 * 1), name='squ_09_w11e')
        self.params['squ_09_w33e'] = He(shape=(e_09, s_09, 3, 3), fan_in=(e_09 * 3 * 3), name='squ_09_w33e')
        self.params['squ_09_g'] = normal(shape=(2 * e_09,), mean=1.0, std_dev=0.02, name='squ_09_g')
        self.params['squ_09_b'] = constant(shape=(2 * e_09,), c=0., name='squ_09_b')
        self.params['squ_10_w11s'] = He(shape=(s_10, 2 * e_09, 1, 1), fan_in=(s_10 * 1 * 1), name='squ_10_w11s')
        self.params['squ_10_w11e'] = He(shape=(e_10, s_10, 1, 1), fan_in=(e_10 * 1 * 1), name='squ_10_w11e')
        self.params['squ_10_w33e'] = He(shape=(e_10, s_10, 3, 3), fan_in=(e_10 * 3 * 3), name='squ_10_w33e')
        self.params['squ_10_w11b'] = He(shape=(2 * e_10, 2 * e_09, 1, 1), fan_in=(2 * e_10 * 1 * 1), name='squ_10_w11b')
        self.params['squ_10_g'] = normal(shape=(2 * e_10,), mean=1.0, std_dev=0.02, name='squ_10_g')
        self.params['squ_10_b'] = constant(shape=(2 * e_10,), c=0., name='squ_10_b')
        self.params['squ_12_w11s'] = He(shape=(s_12, 2 * e_10, 1, 1), fan_in=(s_12 * 1 * 1), name='squ_12_w11s')
        self.params['squ_12_w11e'] = He(shape=(e_12, s_12, 1, 1), fan_in=(e_12 * 1 * 1), name='squ_12_w11e')
        self.params['squ_12_w33e'] = He(shape=(e_12, s_12, 3, 3), fan_in=(e_12 * 3 * 3), name='squ_12_w33e')
        self.params['squ_12_g'] = normal(shape=(2 * e_12,), mean=1.0, std_dev=0.02, name='squ_12_g')
        self.params['squ_12_b'] = constant(shape=(2 * e_12,), c=0., name='squ_12_b')
        self.params['squ_13_conv'] = He(shape=(200, 2 * e_12, 1, 1), fan_in=(200 * 1 * 1), name='squ_13_conv')
        self.params['squ_13_g'] = normal(shape=(200,), mean=1.0, std_dev=0.02, name='squ_13_g')
        self.params['squ_13_b'] = constant(shape=(200,), c=0., name='squ_13_b')

        # ---------
        # Generator
        # ---------

        # Convolutions

        self.params['gen_emb_W'] = He((4800, 128), name='gen_emb_W', fan_in=4800)
        self.params['gen_emb_b'] = constant(shape=(128,), name='gen_emb_b')

        self.params['gen_00_squ_W'] = He((200, 128), name='gen_00_squ_W', fan_in=200)
        self.params['gen_00_squ_b'] = constant(shape=(128,), c=0., name='gen_00_squ_b')

        # Deconvolutions
        self.params['gen_03_W'] = He(hp['gen_03_f'], name='gen_03_W', fan_in=np.prod(hp['gen_03_f'][1:]))
        self.params['gen_03_b'] = constant((hp['gen_03_f'][1],), name='gen_03_b')

        self.params['gen_04_W'] = He(hp['gen_04_f'], name='gen_04_W', fan_in=np.prod(hp['gen_04_f'][1:]))
        self.params['gen_04_b'] = constant((hp['gen_04_f'][1],), name='gen_04_b')

        self.params['gen_05_W'] = He(hp['gen_05_f'], name='gen_05_W', fan_in=np.prod(hp['gen_05_f'][1:]))
        self.params['gen_05_b'] = constant((hp['gen_05_f'][1],), name='gen_05_b')

        self.params['gen_06_W'] = He(hp['gen_06_f'], name='gen_06_W', fan_in=np.prod(hp['gen_06_f'][1:]))
        self.params['gen_06_b'] = constant((hp['gen_06_f'][1],), name='gen_06_b')

        # -------------
        # Discriminator
        # -------------

        dis_f = hp['dis_f']             # dis_f = 128

        self.params['dis_emb_W'] = He((4800, 128), name='dis_emb_W', fan_in=4800)
        self.params['dis_emb_b'] = constant(shape=(128,), name='dis_emb_b')

        self.params['dis_00_W'] = He(shape=(dis_f, 3, 5, 5), fan_in=(dis_f * 5 * 5), name='dis_00_W')
        self.params['dis_00_b'] = constant(shape=(dis_f,), name='dis_00_b')

        self.params['dis_01_W'] = He(shape=(dis_f * 2, dis_f, 5, 5), fan_in=(dis_f * 5 * 5), name='dis_01_W')
        self.params['dis_01_b'] = constant(shape=(dis_f * 2,), name='dis_01_b')
        self.params['dis_01_g'] = normal(shape=(dis_f * 2,), mean=1.0, std_dev=0.02, name='dis_01_g')
        self.params['dis_01_bb'] = constant(shape=(dis_f * 2,), name='dis_01_bb')

        self.params['dis_02_W'] = He(shape=(dis_f * 4, dis_f * 2, 5, 5), fan_in=(dis_f * 2 * 5 * 5), name='dis_02_W')
        self.params['dis_02_b'] = constant(shape=(dis_f * 4,), name='dis_02_b')
        self.params['dis_02_g'] = normal(shape=(dis_f * 4,), mean=1.0, std_dev=0.02, name='dis_02_g')
        self.params['dis_02_bb'] = constant(shape=(dis_f * 4,), name='dis_02_bb')

        self.params['dis_03_W'] = He(shape=(dis_f * 8, dis_f * 4, 5, 5), fan_in=(dis_f * 4 * 5 * 5), name='dis_03_W')
        self.params['dis_03_b'] = constant(shape=(dis_f * 8,), name='dis_03_b')

        self.params['dis_03_mb_W'] = normal(shape=(dis_f * 8 * 4 * 4, 100 * 5), mean=1.0, std_dev=0.02,
                                            name='dis_03_mb_W')
        self.params['dis_03_mb_b'] = constant(shape=(100,), name='dis_03_mb_b')

        self.params['dis_04_g'] = normal(shape=(dis_f * 8 + 100 + 128,), mean=1.0, std_dev=0.02, name='dis_04_g')
        self.params['dis_04_bb'] = constant(shape=(dis_f * 8 + 100 + 128,), name='dis_04_bb')

        self.params['dis_05_W'] = He(shape=((dis_f * 8 + 100 + 128) * 4 * 4, 1), fan_in=((dis_f * 8 * 100) * 4 * 4),
                                     name='dis_05_W')
        self.params['dis_05_b'] = constant(shape=(1,), name='dis_05_b')

        p = self.params

        # -----------------
        # Historical Avg
        # -----------------

        # Generator - Historical Averaging
        list_gen_params = [param for param in self.params.keys() if param.startswith('gen_')]
        g_avg_updates = []
        for gen_param in list_gen_params:
            self.params['avg_' + gen_param] = shared0s(self.params[gen_param].get_value().shape)
            g_avg_updates.append((self.params['avg_' + gen_param],
                                  hist_alpha * self.params[gen_param] +
                                  (1. - hist_alpha) * self.params['avg_' + gen_param]))

        # Discriminator - Historical Averaging
        list_disc_params = [param for param in self.params.keys() if param.startswith('dis_')]
        d_avg_updates = []
        for disc_param in list_disc_params:
            self.params['avg_' + disc_param] = shared0s(self.params[disc_param].get_value().shape)
            d_avg_updates.append((self.params['avg_' + disc_param],
                                  hist_alpha * self.params[disc_param] +
                                  (1. - hist_alpha) * self.params['avg_' + disc_param]))

        # ---------------------
        # SqueezeNet fire modules (with bypass)
        # ---------------------

        def fire_type_A(input, w11s, w11e, w33e, w11b, g, b):
            squ_1x1 = convSqu(input, w11s, subsample=(1, 1), border_mode='half')
            exp_1x1 = convSqu(squ_1x1, w11e, subsample=(1, 1), border_mode='half')
            exp_3x3 = convSqu(squ_1x1, w33e, subsample=(1, 1), border_mode='half')
            exp_con = concat([exp_1x1, exp_3x3], axis=1)
            byp_1x1 = convSqu(input, w11b, subsample=(1, 1), border_mode='half')
            out = relu(exp_con + byp_1x1)
            return batchnorm(out, g=g, b=b)

        def fire_type_B(input, w11s, w11e, w33e, g, b):
            squ_1x1 = convSqu(input, w11s, subsample=(1, 1), border_mode='half')
            exp_1x1 = convSqu(squ_1x1, w11e, subsample=(1, 1), border_mode='half')
            exp_3x3 = convSqu(squ_1x1, w33e, subsample=(1, 1), border_mode='half')
            exp_con = concat([exp_1x1, exp_3x3], axis=1)
            out = relu(exp_con + input)
            return batchnorm(out, g=g, b=b)

        # SqueezeNet
        def squeeze_net(masked_img):
            s_h0 = masked_img  # Image with mask cropped
            # s_h0  = ifelse(already_cropped, X, s_h0)                                                             # b, 3, 64, 64
            s_h1 = convSqu(s_h0, p['squ_01_conv'], subsample=(1, 1), border_mode='half')  # b, 96, 64, 64
            s_h1 = batchnorm(s_h1, g=p['squ_01_g'], b=p['squ_01_b'])
            s_h2 = avg_pool(s_h1, ws=(2, 2))  # b, 96, 32, 32
            s_h3 = fire_type_A(s_h2, *[p['squ_03_' + k] for k in
                                       ['w11s', 'w11e', 'w33e', 'w11b', 'g', 'b']])  # b, 128, 32, 32
            s_h4 = fire_type_B(s_h3, *[p['squ_04_' + k] for k in ['w11s', 'w11e', 'w33e', 'g', 'b']])  # b, 128, 32, 32
            s_h5 = fire_type_A(s_h4, *[p['squ_05_' + k] for k in
                                       ['w11s', 'w11e', 'w33e', 'w11b', 'g', 'b']])  # b, 256, 32, 32
            s_h6 = avg_pool(s_h5, ws=(2, 2))  # b, 256, 16, 16
            s_h7 = fire_type_B(s_h6, *[p['squ_07_' + k] for k in ['w11s', 'w11e', 'w33e', 'g', 'b']])  # b, 256, 16, 16
            s_h8 = fire_type_A(s_h7, *[p['squ_08_' + k] for k in
                                       ['w11s', 'w11e', 'w33e', 'w11b', 'g', 'b']])  # b, 384, 16, 16
            s_h9 = fire_type_B(s_h8, *[p['squ_09_' + k] for k in ['w11s', 'w11e', 'w33e', 'g', 'b']])  # b, 384, 16, 16
            s_h10 = fire_type_A(s_h9, *[p['squ_10_' + k] for k in
                                        ['w11s', 'w11e', 'w33e', 'w11b', 'g', 'b']])  # b, 512, 16, 16
            s_h11 = avg_pool(s_h10, ws=(2, 2))  # b, 512, 8, 8
            s_h12 = fire_type_B(s_h11, *[p['squ_12_' + k] for k in ['w11s', 'w11e', 'w33e', 'g', 'b']])  # b, 512, 8, 8
            s_h12 = ifelse(in_training, dropout(s_h12, in_training, p=0.5), s_h12)  # b, 512, 8, 8
            s_h13 = convSqu(s_h12, p['squ_13_conv'], subsample=(1, 1), border_mode='half')  # b, 200, 8, 8
            s_h13 = batchnorm(s_h13, g=p['squ_13_g'], b=p['squ_13_b'])
            s_x = avg_pool(s_h13, ws=(8, 8)).flatten(2)  # b, 200
            return s_x

        # Generator
        def generator(Z, masked_img, Y, s_x, embedding):
            """ Generator model """

            # dense layer for embedding
            emb_1 = lrelu(T.dot(embedding, p['gen_emb_W']) + p['gen_emb_b'])        # b, 128

            # encode the contour of the image
            # input = masked_img
            gen_00 = lrelu(T.dot(s_x, p['gen_00_squ_W']) + p['gen_00_squ_b'])

            # concatenate the encoded information from mask_img with the noise Z
            prior = concat([Z, emb_1, gen_00]).reshape(shape=(batch_size, 89, 2, 2))

            # Generate X
            # 4, 4, 3
            # TODO: add dropout
            #prior = dropout(prior, in_training, p=hp['dropout_p'])

            gen_03 = lrelu(batchnorm(deconv(prior, hp['gen_03_f'], p['gen_03_W'], hp['gen_03_i'],
                              (2, 2))) + p['gen_03_b'].dimshuffle('x', 0, 'x', 'x'))  # f = (2, 2)

            # 8, 8, 3
            gen_04 = lrelu(batchnorm(deconv(gen_03, hp['gen_04_f'], p['gen_04_W'], hp['gen_04_i'],
                         (2, 2))) + p['gen_04_b'].dimshuffle('x', 0, 'x', 'x'))  # f = (2, 2)
            #gen_04 = dropout(gen_04, in_training, p=hp['dropout_p'])



            # 16, 16, 3
            gen_05 = lrelu(batchnorm(deconv(gen_04, hp['gen_05_f'], p['gen_05_W'], hp['gen_05_i'],
                              (2, 2))) + p['gen_05_b'].dimshuffle('x', 0, 'x', 'x'))  # f = (2, 2)


            # 32, 32, 3
            gen_06 = deconv(gen_05, hp['gen_06_f'], p['gen_06_W'], hp['gen_06_i'],
                         (2, 2)) + p['gen_06_b'].dimshuffle('x', 0, 'x', 'x')  # f = (2, 2)

            gen_06 = T.tanh(gen_06)
            #gen_07 = gen_06 * mask + (1 - mask) * Y
            masked_img = T.set_subtensor(masked_img[:, :, 16:48, 16:48], gen_06)
            gen_07 = masked_img
            return gen_07

        # Discriminator
        # Adapted from https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py#L132
        def minibatch(d, W, b):
            """ Minibatch Discrimination """
            mb_h0 = T.dot(d.flatten(2), W)  # b, 500
            mb_h0 = mb_h0.reshape((d.shape[0], hp['mb_nb_kernels'], hp['mb_kernel_dim']))  # b, 100, 5
            mb_h1 = mb_h0.dimshuffle(0, 1, 2, 'x') - mb_h0.dimshuffle('x', 1, 2, 0)
            mb_h1 = T.sum(abs(mb_h1), axis=2) + 1e6 * T.eye(d.shape[0]).dimshuffle(0, 'x', 1)
            mb_h2 = T.sum(T.exp(-mb_h1), axis=2) + b  # b, nb_kernel
            mb_h2 = mb_h2.dimshuffle(0, 1, 'x', 'x')
            mb_h2 = T.repeat(mb_h2, 4, axis=2)
            mb_h2 = T.repeat(mb_h2, 4, axis=3)
            return mb_h2            # 64, 100, 4, 4

        def discriminator(X, Y, embedding):

            emb_1 = lrelu(T.dot(embedding, p['dis_emb_W']) + p['dis_emb_b']).dimshuffle(0, 1, 'x', 'x')     # b, 128
            emb_1 = T.repeat(emb_1, 4, axis=2)
            emb_1 = T.repeat(emb_1, 4, axis=3)
            images = X * mask + (1. - mask) * (Y + X) / 2.
            dis_00 = lrelu(conv(images, (dis_f, 3, 5, 5), p['dis_00_W'], subsample=(2, 2), border_mode=(2, 2))
                           + p['dis_00_b'].dimshuffle('x', 0, 'x', 'x'))

            dis_01 = conv(dis_00, (dis_f * 2, dis_f, 5, 5), p['dis_01_W'], subsample=(2, 2), border_mode=(2, 2)) \
                     + p['dis_01_b'].dimshuffle('x', 0, 'x', 'x')
            dis_01 = lrelu(batchnorm(dis_01, g=p['dis_01_g'], b=p['dis_01_bb']))
            #dis_01 = dropout(dis_01, in_training, p=hp['dropout_p'])
            dis_02 = conv(dis_01, (dis_f * 4, dis_f * 2, 5, 5), p['dis_02_W'], subsample=(2, 2), border_mode=(2, 2)) \
                     + p['dis_02_b'].dimshuffle('x', 0, 'x', 'x')
            dis_02 = lrelu(batchnorm(dis_02, g=p['dis_02_g'], b=p['dis_02_bb']))

            dis_03 = conv(dis_02, (dis_f * 8, dis_f * 4, 5, 5), p['dis_03_W'], subsample=(2, 2), border_mode=(2, 2)) \
                     + p['dis_03_b'].dimshuffle('x', 0, 'x', 'x')
            dis_03_mb = minibatch(dis_03, W=p['dis_03_mb_W'], b=p['dis_03_mb_b'])

            dis_04 = concat([dis_03, emb_1, dis_03_mb], axis=1)               # join (64, 1024, 4, 4) dis_03
                                                                                                    #      (64, 128, 1, 1)  emb_1.dimshuffle(0,1,'x','x')
            dis_04 = lrelu(batchnorm(dis_04, g=p['dis_04_g'], b=p['dis_04_bb']))                    #      (64, 100, 4, 4)  dis_03_mb
                                                                                                    # batchnorm(dis_04).shape = (64, 1252,1 ,1)
            dis_04 = T.flatten(dis_04, 2)                                                           # flatten => (64, 1252)

            dis_Y = T.nnet.sigmoid(T.dot(dis_04, p['dis_05_W']) + p['dis_05_b'])

            return dis_Y, dis_02

        s_x = squeeze_net(masked_img)

        gX = generator(Z, masked_img, Y, s_x, embedding)

        p_real, real_dis_02 = discriminator(Y, Y, embedding)
        p_gen, gen_dis_02 = discriminator(gX, Y, embedding)

        g_feature_matching = MeanSquaredError(gen_dis_02, real_dis_02)

        # historical averaging
        #g_avg_cost, d_avg_cost = 0., 0.
        #nb_g_avg_param, nb_d_avg_param = 0., 0.
        #for param in self.params.keys():
        #    if param.startswith('gen_'):
        #        g_avg_cost += MeanSquaredError(self.params[param], self.params['avg_' + param])
        #        nb_g_avg_param += 1
        #    if param.startswith('dis_'):
        #        d_avg_cost += MeanSquaredError(self.params[param], self.params['avg_' + param])
        #        nb_d_avg_param += 1
        #g_avg_cost = g_avg_cost / max(1., nb_g_avg_param)
        #d_avg_cost = d_avg_cost / max(1., nb_d_avg_param)

        d_cost_real = BinaryCrossEntropy(p_real, T.ones(p_real.shape)).mean()
        d_cost_gen = BinaryCrossEntropy(p_gen, T.zeros(p_gen.shape)).mean()
        g_cost_d = BinaryCrossEntropy(p_gen, T.ones(p_gen.shape)).mean()
        #g_border_cost = MeanSquaredError(gX * mask, Y * mask)


        # TODO: test border cost / hist. averaging -- probably want to remove those 2
        #d_cost = d_cost_real + d_cost_gen + d_avg_cost
        #g_cost = g_cost_d + g_avg_cost + g_border_cost + g_feature_matching
        d_cost = d_cost_real + d_cost_gen
        g_cost = g_cost_d + g_feature_matching

        #cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen, g_border_cost]

        cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

        # TODO: switch d_updater for SGD
        #d_updater = Adam(lr=hp['learning_rate'], b1=adam_b1, regularizer=Regularizer(l2=l2_reg), clipnorm=10.)
        d_updater = SGD(lr=hp['learning_rate'], regularizer=Regularizer(l2=l2_reg), clipnorm=10.)
        g_updater = Adam(lr=hp['learning_rate'], b1=adam_b1, regularizer=Regularizer(l2=l2_reg), clipnorm=10.)

        d_params = [self.params[k] for k in self.params.keys()
                    if k.startswith('dis_')]
        d_updates = d_updater(d_params, d_cost)

        g_params = [self.params[k] for k in self.params.keys() if k.startswith('gen_')]
        g_params += [self.params[k] for k in self.params.keys() if k.startswith('squ_')]


        g_updates = g_updater(g_params, g_cost)
        # g_border_updates = g_updater(g_params, g_border_cost)
        d_g_updates = d_updates + g_updates  # + g_border_updates

        # -------------------
        # Compiling Functions
        # -------------------

        self.generate_image = theano.function(
            [index, Z],
            [gX],
            givens={
                masked_img: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size],
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size],
                embedding: self.gpu_dataset_emb[index * batch_size: (index + 1) * batch_size],
                in_training: np.cast['int8'](0)
            },
            on_unused_input='warn'
        )


        print('Compiling train_d_g_fn()...')
        self.train_g_d_fn = theano.function(
            inputs=[index, Z],
            outputs=cost,
            updates=d_g_updates,
            givens={
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size],
                masked_img: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size],
                embedding: self.gpu_dataset_emb[index * batch_size: (index + 1) * batch_size],
                in_training: np.cast['int8'](1)
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
        Z_dim = hp['Z_dim']

        learning_rate_epoch = hp['learning_rate_epoch']
        learning_rate_adj = hp['learning_rate_adj']
        initial_learning_rate = hp['learning_rate']
        last_cost_g = []
        last_cost_d = []

        DONE_LOOPING = False
        epoch = 0
        #self.params['hyper_learning_rate'].set_value(initial_learning_rate)
        batch_idx = 0
        while epoch < nb_epochs and not DONE_LOOPING:
            epoch += 1
            masterbatch_idx = 0

            # loop through the GPU batches
            for gpu_batch in iterate_minibatches(batch_size * nb_batch_store_gpu, 'train2014'):

                self.gpu_dataset_X.set_value(gpu_batch[0])
                self.gpu_dataset_Y.set_value(gpu_batch[1])
                self.gpu_dataset_emb.set_value(gpu_batch[2])

                for idx in range(nb_batch_store_gpu):
                    batch_idx += 1
                    Z = rng.normal(0, 1, size=(batch_size, Z_dim)).astype('float32')
                    g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen = self.train_g_d_fn(idx, Z)

                    last_cost_g.append(g_cost)
                    last_cost_g = last_cost_g[-1000:]
                    last_cost_d.append(d_cost)
                    last_cost_d = last_cost_d[-1000:]
                    print('Batch %05d/%05d - Epoch %04d -- Loss G/D: %.4f/%.4f ' %
                          (batch_idx, nb_batch_train, epoch, g_cost, d_cost))
                    # TODO: copy loss to some file
                    f = open('loss.txt', 'w')
                    f.write('%06d, %.4f, %.4f \n' % (batch_idx + 15 * 645, g_cost, d_cost))
                    #f.close()

                    #if batch_idx % 100 == 0:
                    #    self.save_img(Z, 0)
                        #self.save('model_gan_01.pkl')
            if epoch % 5 == 0:
                self.save('model1_gan_do_epoch_%04d.pkl' % (epoch))

    def save(self, file_path):
        """ Save the model """
        f = open(file_path, 'wb')
        cPickle.dump((self.params.values(), self.hparams.values()), f)
        f.close()

    def generate(self, Z, batch_idx):
        data = load_data(batch_idx, 128, 'val2014')
        self.gpu_dataset_Y.set_value(data[1])
        self.gpu_dataset_X.set_value(data[0])
        self.gpu_dataset_emb.set_value(data[2])
        imgs = self.generate_image(0, Z)
        return imgs, data[1]

    def save_img(self, Z, batch_idx):
        """ Save an image """
        img = self.generate(Z, batch_idx)
        i = ((img[0][0][2].transpose((1, 2, 0)) + 1.) / 2.) * 255.
        i2 = ((img[0][0][1].transpose((1, 2, 0)) + 1.) / 2.) * 255.
        i3 = ((img[0][0][3].transpose((1, 2, 0)) + 1.) / 2.) * 255.
        Image.fromarray(i.astype('uint8')).show()
        Image.fromarray(i2.astype('uint8')).show()
        Image.fromarray(i3.astype('uint8')).show()

    def load(self, file_path):
        """ Load model """
        f = open(file_path, 'rb')
        params = cPickle.load(f)
        f.close()
        params = params[0]
        print('loading model %s...' % file_path)

        keys = [key for key in self.params.keys()]
        values = [value for value in params]
        assert len(keys) == len(values)
        for i in range(len(keys)):
            self.params[keys[i]].set_value(values[i].get_value())
        return params

    def show_imgs(self, Z, batch_idx):
        imgs = self.generate(Z, batch_idx)
        for img in imgs[0][0]:
            Image.fromarray((((img.transpose((1, 2, 0)) + 1.)
                              / 2.) * 255.).astype('uint8'), 'RGB').show()
        # TODO: add image id's to some file to capture captions

    def generate_img(self, Z, gpu_dataset_X, gpu_dataset_emb, gpu_dataset_Y):

        self.gpu_dataset_X.set_value(gpu_dataset_X.get_value())
        self.gpu_dataset_Y.set_value(gpu_dataset_Y.get_value())
        self.gpu_dataset_emb.set_value(gpu_dataset_emb.get_value())

        for idx in range(5):
            if idx == 0:
                imgs = self.generate_image(idx, Z[idx])
            else:
                imgs = np.concatenate((imgs, self.generate_image(idx, Z[idx])), axis=0)
        return imgs.reshape(( 320, 3, 64, 64))

    def get_hparams(self):
        return {
            "batch_size": 128,
            "nb_batch_store_gpu": 5,  # # of batch to store on the GPU
            "k": 1,  # # of discrim updates for each gen update
            "l2": 2.5e-5,  # l2 weight decay
            "b1": 0.5,  # momentum term of adam
            "learning_rate": 0.0001,  # initial learning rate for adam
            'learning_rate_epoch': 10,
            'learning_rate_adj': 0.8,
            'mb_kernel_dim': 5,
            'mb_nb_kernels': 100,
            'Z_dim': 100,
            'nb_epochs': 500,
            'dis_f': 128,
            'dropout_p': 0.3,

            # hyper-parameters for the model

            # ---------
            # Generator
            # ---------

            # Convolutions
            'gen_00_f': (64, 3, 5, 5),
            'gen_01_f': (64, 64, 5, 5),
            'gen_02_f': (64, 64, 5, 5),

            # Deconvolutions
            'gen_03_f': (89, 64, 2, 2),
            'gen_03_i': (128, 64, 4, 4),  # b, 64, 6, 6 after unpool

            'gen_04_f': (64, 64, 2, 2),
            'gen_04_i': (128, 64, 8, 8),  # b, 64, 14, 14 after unpool

            'gen_05_f': (64, 64, 2, 2),
            'gen_05_i': (128, 64, 16, 16),  # b, 64, 30, 30

            'gen_06_f': (64, 3, 2, 2),
            'gen_06_i': (128, 3, 32, 32),       #b, 3, 32, 32


            # -------------
            # Discriminator
            # -------------

            # Convolutions
            'dis_00_f': (256, 3, 5, 5),
            'dis_01_f': (512, 256, 5, 5),
            'dis_02_f': (1024, 512, 5, 5),

            # Minibatch Discrimination

            'dis_03_f': (1024 * 4 * 4, 500),
            'dis_04_f': (1600, 1)
            # 'dis_04_f': (1024 * 4 * 4, 1)

        }


if __name__ == '__main__':
    network = GenerativeAdversarialNetwork()
    network.build_model()
    network.load('model_gan_do_epoch_0090.pkl')
    network.train_model()




























