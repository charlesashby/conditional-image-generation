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

theano.config.exception_verbosity = 'high'

class BEGAN(object):
    """ Deep Residual GAN Implementation """

    def build_model(self):

        print('... Building Model')

        self.hparams = self.get_hparams()
        hp = self.hparams

        adam_b1 = 0.7
        l2_reg = 1e-5
        batch_size = hp['batch_size']
        nb_batch_store_gpu = hp['nb_batch_store_gpu']
        nb_batch_train = 82611 // batch_size

        Z = T.matrix('Z', dtype='float32')
        Y = T.tensor4(name='Y', dtype='float32')
        embedding = T.matrix(name='embed', dtype='float32')
        masked_img = T.tensor4(name='mask_img', dtype='float32')
        index = T.iscalar('index')
        in_training = T.bscalar('in_training')

        # space for storing dataset on GPU
        self.gpu_dataset_X = shared0s(shape=(batch_size * nb_batch_store_gpu, 3, 64, 64))
        self.gpu_dataset_Y = shared0s(shape=(batch_size * nb_batch_store_gpu, 3, 64, 64))
        self.gpu_dataset_emb = shared0s(shape=(batch_size * nb_batch_store_gpu, 4800))

        # Define parameters
        self.params = collections.OrderedDict()

        # ----------
        # Generator
        # ----------

        #prior concat
        self.params['k_t'] = theano.shared(0., name='k_t')

        self.params['gen_emb_W'] = He((4800, 128), name='gen_emb_W', fan_in=4800)
        self.params['gen_emb_b'] = constant(shape=(128,), name='gen_emb_b')

        self.params['gen_00_squ_W'] = He((200, 128), name='gen_00_squ_W', fan_in=200)
        self.params['gen_00_squ_b'] = constant(shape=(128,), c=0., name='gen_00_squ_b')

        self.params['gen_01_W'] = He((128, 3, 3, 3),name='gen_01_W', fan_in=128*3*3)
        self.params['gen_01_b'] = constant(shape=(128,),name='gen_01_b')




        self.params['gen_02_W'] = He((128 * 2, 128, 3, 3),name='gen_02_W', fan_in=128*3*3)
        self.params['gen_02_b'] = constant(shape=(128 * 2,),name='gen_02_b')


        self.params['gen_03_W'] = He((128 * 3, 128 * 2, 3, 3),name='gen_03_W', fan_in=128*3*3)
        self.params['gen_03_b'] = constant(shape=(128 * 3,),name='gen_03_b')


        self.params['gen_04_W'] = He((128 * 4, 128 * 3, 3, 3),name='gen_04_W', fan_in=128*3*3)
        self.params['gen_04_b'] = constant(shape=(128 * 4,),name='gen_04_b')


        self.params['gen_05_W'] = He((200, 128 * 4, 3, 3),name='gen_05_W', fan_in=200*3*3)
        self.params['gen_05_b'] = constant(shape=(200,),name='gen_05_b')


        self.params['gen_06_W'] = He((128, 89, 3, 3),name='gen_06_W', fan_in=128*3*3)
        self.params['gen_06_b'] = constant(shape=(128,),name='gen_06_b')


        self.params['gen_07_W'] = He((128, 128, 3, 3),name='gen_07_W', fan_in=128*3*3)
        self.params['gen_07_b'] = constant(shape=(128,),name='gen_07_b')
        self.params['gen_07bn_W'] = He((128,), name='gen_07bn_W', fan_in=128)
        self.params['gen_07bn_b'] = constant((128,), name='gen_07bn_b')

        self.params['gen_08_W'] = He((128, 128, 3, 3),name='gen_08_W', fan_in=128*3*3)
        self.params['gen_08_b'] = constant(shape=(128,),name='gen_08_b')
        self.params['gen_08bn_W'] = He((128,), name='gen_08bn_W', fan_in=128)
        self.params['gen_08bn_b'] = constant((128,), name='gen_08bn_b')

        self.params['gen_09_W'] = He((128, 128, 3, 3),name='gen_09_W', fan_in=128*3*3)
        self.params['gen_09_b'] = constant(shape=(128,),name='gen_09_b')
        self.params['gen_09bn_W'] = He((128,), name='gen_09bn_W', fan_in=128)
        self.params['gen_09bn_b'] = constant((128,), name='gen_09bn_b')

        self.params['gen_010_W'] = He((128, 128, 3, 3),name='gen_010_W', fan_in=128*3*3)
        self.params['gen_010_b'] = constant(shape=(128,),name='gen_010_b')
        #self.params['gen_010bn_W'] = He((128,), name='gen_010bn_W', fan_in=128)
        #self.params['gen_010bn_b'] = constant((128,), name='gen_010bn_b')

        self.params['gen_011_W'] = He((3, 128, 3, 3),name='gen_011_W', fan_in=3*3*3)
        self.params['gen_011_b'] = constant(shape=(3,),name='gen_011_b')


        # -------------
        # Discriminator
        # -------------

        self.params['dis_emb_W'] = He((4800, 128), name='dis_emb_W', fan_in=4800)
        self.params['dis_emb_b'] = constant(shape=(128,), name='dis_emb_b')

        self.params['dis_00_squ_W'] = He((200, 128), name='dis_00_squ_W', fan_in=200)
        self.params['dis_00_squ_b'] = constant(shape=(128,), c=0., name='dis_00_squ_b')

        self.params['dis_01_W'] = He((128, 3, 3, 3), name='dis_01_W', fan_in=128 * 3 * 3)
        self.params['dis_01_b'] = constant(shape=(128,), name='dis_01_b')


        self.params['dis_02_W'] = He((128 * 2, 128, 3, 3), name='dis_02_W', fan_in=128 * 3 * 3)
        self.params['dis_02_b'] = constant(shape=(128 * 2,), name='dis_02_b')


        self.params['dis_03_W'] = He((128 * 3, 128 * 2, 3, 3), name='dis_03_W', fan_in=128 * 3 * 3)
        self.params['dis_03_b'] = constant(shape=(128 * 3,), name='dis_03_b')


        self.params['dis_04_W'] = He((128 * 4, 128 * 3, 3, 3), name='dis_04_W', fan_in=128 * 3 * 3)
        self.params['dis_04_b'] = constant(shape=(128 * 4,), name='dis_04_b')


        self.params['dis_05_W'] = He((200, 128 * 4, 3, 3), name='dis_05_W', fan_in=200 * 3 * 3)
        self.params['dis_05_b'] = constant(shape=(200,), name='dis_05_b')


        self.params['dis_06_W'] = He((128, 89, 3, 3), name='dis_06_W', fan_in=128 * 3 * 3)
        self.params['dis_06_b'] = constant(shape=(128,), name='dis_06_b')


        self.params['dis_07_W'] = He((128, 128, 3, 3), name='dis_07_W', fan_in=128 * 3 * 3)
        self.params['dis_07_b'] = constant(shape=(128,), name='dis_07_b')
        self.params['dis_07bn_W'] = He((128,), name='dis_07bn_W', fan_in=128)
        self.params['dis_07bn_b'] = constant((128,), name='dis_07bn_b')

        self.params['dis_08_W'] = He((128, 128, 3, 3), name='dis_08_W', fan_in=128 * 3 * 3)
        self.params['dis_08_b'] = constant(shape=(128,), name='dis_08_b')
        self.params['dis_08bn_W'] = He((128,), name='dis_08bn_W', fan_in=128)
        self.params['dis_08bn_b'] = constant((128,), name='dis_08bn_b')

        self.params['dis_09_W'] = He((128, 128, 3, 3), name='dis_09_W', fan_in=128 * 3 * 3)
        self.params['dis_09_b'] = constant(shape=(128,), name='dis_09_b')
        self.params['dis_09bn_W'] = He((128,), name='dis_09bn_W', fan_in=128)
        self.params['dis_09bn_b'] = constant((128,), name='dis_09bn_b')

        self.params['dis_010_W'] = He((128, 128, 3, 3), name='dis_010_W', fan_in=128 * 3 * 3)
        self.params['dis_010_b'] = constant(shape=(128,), name='dis_010_b')
        #self.params['dis_010bn_W'] = He((128,), name='dis_010bn_W', fan_in=128)
        #self.params['dis_010bn_b'] = constant((128,), name='dis_010bn_b')

        self.params['dis_011_W'] = He((3, 128, 3, 3), name='dis_011_W', fan_in=3 * 3 * 3)
        self.params['dis_011_b'] = constant(shape=(3,), name='dis_011_b')

        p = self.params




        def _encoderG(masked_img):
            #if in_training:
            #masked_img = dropout(masked_img, in_training, p=0.2)

            # b, 128, 31, 31
            conv1 = elu(conv(masked_img, (128, 3, 3, 3), p['gen_01_W'], subsample=(2,2)) +
                        p['gen_01_b'].dimshuffle('x', 0, 'x', 'x'))
            # b, 128, 15, 15
            conv2 = elu(conv(conv1, (128 * 2, 128, 3, 3), p['gen_02_W'], subsample=(2,2)) +
                        p['gen_02_b'].dimshuffle('x', 0, 'x', 'x'))

            #conv2 = dropout(conv2, in_training, p=0.5)
            # b, 128, 7, 7
            conv3 = elu(conv(conv2, (128 * 3, 128 * 2, 3, 3), p['gen_03_W'], subsample=(2, 2)) +
                        p['gen_03_b'].dimshuffle('x', 0, 'x', 'x'))
            #if in_training:
            #    conv3 = dropout(conv3)

            # b, 128, 3, 3
            conv4 = elu(conv(conv3, (128 * 4, 128 * 3, 3, 3), p['gen_04_W'], subsample=(2,2)) +
                        p['gen_04_b'].dimshuffle('x', 0, 'x', 'x'))

            #conv4 = dropout(conv4, in_training, p=0.5)
            # b, 200, 1, 1
            conv5 = elu(conv(conv4, (200, 128 * 4, 3, 3), p['gen_05_W'], subsample=(2,2)) +
                        p['gen_05_b'].dimshuffle('x', 0, 'x', 'x'))
            return conv5.flatten(2)


        def _decoderG(Z, masked_img, e_x, embedding):
            """ Generator model """

            # dense layer for embedding
            emb_1 = T.dot(embedding, p['gen_emb_W']) + p['gen_emb_b']      # b, 128

            # encode the contour of the image
            # input = masked_img
            gen_00 = T.dot(e_x, p['gen_00_squ_W']) + p['gen_00_squ_b']

            # concatenate the encoded information from mask_img with the noise Z
            # b, 100 + 128 + 128 => b, 89, 2, 2
            prior = concat([Z, emb_1, gen_00]).reshape(shape=(batch_size, 89, 2, 2))

            #prior = dropout(prior, in_training, p=0.2)
            #if in_training:
            #    prior = dropout(prior)

            # b, 89, 6, 6
            layer6 = unpool(prior, (3,3))
            # b, 128, 4, 4
            layer6 = elu(conv(layer6, (128, 89, 3, 3), p['gen_06_W']) +
                         p['gen_06_b'].dimshuffle('x', 0, 'x', 'x'))
            # b, 128, 8, 8
            layer7 = unpool(layer6, (2,2))
            # b, 128, 6, 6
            layer7 = elu(batchnorm(conv(layer7, (128, 128, 3, 3), p['gen_07_W']),
                            g=p['gen_07bn_W'], b=p['gen_07bn_b']) + p['gen_07_b'].dimshuffle('x', 0, 'x', 'x'))

            #layer7 = dropout(layer7, in_training, p=0.2)
            # b, 128, 12, 12
            layer8 = unpool(layer7, (2,2))
            # b, 128, 10, 10
            layer8 = elu(batchnorm(conv(layer8, (128, 128, 3, 3), p['gen_08_W']),
                            g=p['gen_08bn_W'], b=p['gen_08bn_b']) + p['gen_08_b'].dimshuffle('x', 0, 'x', 'x'))
            #if in_training:
            #    layer8 = dropout(layer8)
            # b, 128, 20, 20
            layer9 = unpool(layer8, (2,2))
            # b, 128, 18, 18
            layer9 = elu(batchnorm(conv(layer9, (128, 128, 3, 3), p['gen_09_W']),
                            g=p['gen_09bn_W'], b=p['gen_09bn_b']) + p['gen_09_b'].dimshuffle('x', 0, 'x', 'x'))

            #layer9 = dropout(layer9, in_training, p=0.2)

            # b, 128, 36, 36
            layer10 = unpool(layer9, (2,2))
            # b, 128, 34, 34
            layer10 = elu(conv(layer10, (128, 128, 3, 3), p['gen_010_W']) +
                          p['gen_010_b'].dimshuffle('x', 0, 'x', 'x'))
            # b, 3, 32, 32
            layer11 = conv(layer10, (3, 128, 3, 3), p['gen_011_W']) + p['gen_011_b'].dimshuffle('x', 0, 'x', 'x')

            generated_img = T.set_subtensor(masked_img[:, :, 16:48, 16:48], layer11)

            return generated_img, layer10

        def _encoderD(masked_img):
            #if in_training:
            #masked_img = dropout(masked_img, in_training,p=0.2)
            # b, 128, 31, 31

            conv1 = elu(conv(masked_img, (128, 3, 3, 3), p['dis_01_W'], subsample=(2, 2)) +
                        p['dis_01_b'].dimshuffle('x', 0, 'x', 'x'))
            # b, 128, 15, 15
            conv2 = elu(conv(conv1, (128 * 2, 128, 3, 3), p['dis_02_W'], subsample=(2, 2)) +
                        p['dis_02_b'].dimshuffle('x', 0, 'x', 'x'))

            #conv2 = dropout(conv2, in_training, p=0.5)
            # b, 128, 7, 7
            conv3 = elu(conv(conv2, (128 * 3, 128 * 2, 3, 3), p['dis_03_W'], subsample=(2, 2)) +
                        p['dis_03_b'].dimshuffle('x', 0, 'x', 'x'))
            #if in_training:
            #    conv3 = dropout(conv3)
            # b, 128, 3, 3
            conv4 = elu(conv(conv3, (128 * 4, 128 * 3, 3, 3), p['dis_04_W'], subsample=(2, 2)) +
                        p['dis_04_b'].dimshuffle('x', 0, 'x', 'x'))

            #conv4 = dropout(conv4, in_training, p=0.5)
            # b, 200, 1, 1
            conv5 = elu(conv(conv4, (200, 128 * 4, 3, 3), p['dis_05_W'], subsample=(2, 2)) +
                        p['dis_05_b'].dimshuffle('x', 0, 'x', 'x'))
            return conv5.flatten(2)

        def _decoderD(Z, masked_img, e_x, embedding):
            """ Generator model """

            # dense layer for embedding
            emb_1 = elu(T.dot(embedding, p['dis_emb_W']) + p['dis_emb_b'])  # b, 128

            # encode the contour of the image
            # input = masked_img
            dis_00 = T.dot(e_x, p['dis_00_squ_W']) + p['dis_00_squ_b']

            # concatenate the encoded information from mask_img with the noise Z
            # b, 100 + 128 + 128 => b, 89, 2, 2
            prior = concat([Z, emb_1, dis_00]).reshape(shape=(batch_size, 89, 2, 2))

            #prior = dropout(prior, in_training, p=0.2)
            #if in_training:
            #    prior = dropout(prior, 0.3)
            # b, 89, 6, 6
            layer6 = unpool(prior, (3, 3))
            # b, 128, 4, 4
            layer6 = elu(conv(layer6, (128, 89, 3, 3), p['dis_06_W']) +
                         p['dis_06_b'].dimshuffle('x', 0, 'x', 'x'))
            # b, 128, 8, 8
            layer7 = unpool(layer6, (2, 2))
            # b, 128, 6, 6
            layer7 = elu(batchnorm(conv(layer7, (128, 128, 3, 3), p['dis_07_W']),
                                     g=p['dis_07bn_W'], b=p['dis_07bn_b']) + p['dis_07_b'].dimshuffle('x', 0, 'x', 'x'))

            #layer7 = dropout(layer7, in_training, p=0.2)
            # b, 128, 12, 12
            layer8 = unpool(layer7, (2, 2))
            # b, 128, 10, 10
            layer8 = elu(batchnorm(conv(layer8, (128, 128, 3, 3), p['dis_08_W']),
                                     g=p['dis_08bn_W'], b=p['dis_08bn_b']) + p['dis_08_b'].dimshuffle('x', 0, 'x','x'))
            #if in_training:
            #    layer8 = dropout(layer8)
            # b, 128, 20, 20
            layer9 = unpool(layer8, (2, 2))
            # b, 128, 18, 18
            layer9 = elu(batchnorm(conv(layer9, (128, 128, 3, 3), p['dis_09_W']),
                                     g=p['dis_09bn_W'], b=p['dis_09bn_b']) + p['dis_09_b'].dimshuffle('x', 0, 'x', 'x'))

            #layer9 = dropout(layer9, in_training, p=0.2)
            # b, 128, 36, 36
            layer10 = unpool(layer9, (2, 2))
            # b, 128, 34, 34
            layer10 = elu(conv(layer10, (128, 128, 3, 3), p['dis_010_W']) +
                          p['dis_010_b'].dimshuffle('x', 0, 'x', 'x'))
            # b, 3, 32, 32
            layer11 = conv(layer10, (3, 128, 3, 3), p['dis_011_W']) + p['dis_011_b'].dimshuffle('x', 0, 'x', 'x')

            generated_img = T.set_subtensor(masked_img[:, :, 16:48, 16:48], layer11)

            return generated_img, layer10

        # The following is taken from the original BEGAN repo:
        # The boundary equilibrium GAN uses an approximation of the Wasserstein loss between
        # the distributions of pixel-wise autoencoder loss based on the discriminator performance
        # on real vs. generated data.

        # This simplifies to reducing the L1 norm of the autoencoder loss:
        # making the discriminator objective to perform well on real images and
        # poorly on generated images; with the generator objective to create samples which the
        # discriminator will perform well upon

        # args:
        #   D_real_in: input to discriminator with real sample
        #   D_real_out: outpur from discriminator with real sample
        #   D_gen_in: input to discriminator with generated sample
        #   D_gen_out: output from discriminator with generated sample
        #   k_t: weighting parameter which constantly updates during training
        #   gamma: diversity ratio, used to control model equilibrium

        # returns:
        #   D_loss: discriminator loss to minimize
        #   G_loss: generator loss to minimize
        #   k_tp: value of k_t for next train step
        #   convergence_measure: measure of model convergence

        D_gen_in, featureG = _decoderG(Z, masked_img, _encoderG(masked_img), embedding) # generate img
        D_gen_out, featureD_G = _decoderD(Z, masked_img, _encoderD(D_gen_in), embedding)    # pass generated img through D

        D_real_in = Y
        D_real_out, featureD = _decoderD(Z, Y, _encoderD(D_real_in), embedding)          # Pass the real img through D

        D_real_out = self.denorm_img(D_real_out)
        D_gen_in = self.denorm_img(D_gen_in)
        D_real_in = self.denorm_img(D_real_in)
        D_gen_out = self.denorm_img(D_gen_out)

        loss_center = T.sqrt(MeanSquaredError(D_gen_in, D_real_in) + 1e-5)

        mu_real = T.mean(AbsoluteValue(D_real_out, D_real_in))  # d_loss_real
        mu_gen = T.mean(AbsoluteValue(D_gen_out, D_gen_in))     # d_loss_fake
        #feature_matching = MeanSquaredError(featureG, featureD)
        d_cost = mu_real - p['k_t'] * mu_gen
        #d_cost = mu_real - mu_gen
        g_cost = mu_gen + 0.25 * loss_center
        lam = 0.005             # learning rate for k
        gamma = 0.50
        k_tp = T.min([T.max([p['k_t'] + lam * (gamma * mu_real - mu_gen), 0.0]), 1.])
        k_update = (p['k_t'], k_tp)
        convergence_measure = mu_real + np.abs(gamma * mu_real - mu_gen)
        gamma_approx = mu_gen / mu_real


        cost = [g_cost, d_cost, k_tp, convergence_measure, gamma_approx, loss_center]
        # TODO: test RMSPROP
        d_updater = Adam(lr=hp['learning_rate'], regularizer=Regularizer(l2=l2_reg), clipnorm=10.)
        #d_updater = SGD(lr=hp['learning_rate'], regularizer=Regularizer(l2=l2_reg), clipnorm=10.)
        g_updater = Adam(lr=hp['learning_rate'], regularizer=Regularizer(l2=l2_reg), clipnorm=10.)

        #d_updater = RMSprop(lr=hp['learning_rate'], regularizer=Regularizer(clamp_u=0.01, clamp_l=-0.01))
        #g_updater = RMSprop(lr=hp['learning_rate'])

        d_params = [self.params[k] for k in self.params.keys()
                    if k.startswith('dis_')]
        d_updates = d_updater(d_params, d_cost)


        g_params = [self.params[k] for k in self.params.keys() if k.startswith('gen_')]

        g_updates = g_updater(g_params, g_cost)


        d_g_updates = d_updates + g_updates + [k_update]

        # -------------------
        # Compiling Functions
        # -------------------

        self.generate_image = theano.function(
            [index, Z],
            [D_gen_in, D_real_out, D_gen_out],
            givens={
                masked_img: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size],
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size],
                embedding: self.gpu_dataset_emb[index * batch_size: (index + 1) * batch_size],
                in_training: np.cast['int8'](0)
            },
            on_unused_input='warn'
        )

        print('... Compiling train_d_g_fn()')
        self.train_g_d_fn = theano.function(
            inputs=[index, Z],
            outputs=cost,
            updates=d_g_updates,
            givens={
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size],
                masked_img: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size],
                embedding: self.gpu_dataset_emb[index * batch_size: (index + 1) * batch_size],
                in_training: np.cast['int8'](1)
            },
            on_unused_input='warn'
        )
        print('... Done Compiling train_d_g_fn()')



    def train_model(self):
        """ Train the GAN """
        print('... Training Model')

        rng = np.random.RandomState(1)
        hp = self.hparams
        batch_size = hp['batch_size']
        nb_batch_store_gpu = hp['nb_batch_store_gpu']
        training_set = 82611

        nb_batch_train = 82611 // batch_size
        nb_epochs = hp['nb_epochs']
        Z_dim = hp['Z_dim']

        DONE_LOOPING = False
        epoch = 0
        batch_idx = 0
        g_iterations = 0

        while epoch < nb_epochs and not DONE_LOOPING:
            epoch += 1


            for gpu_batch in iterate_minibatches(batch_size * nb_batch_store_gpu, 'train2014', training_set):

                self.gpu_dataset_X.set_value(gpu_batch[0])
                self.gpu_dataset_Y.set_value(gpu_batch[1])
                self.gpu_dataset_emb.set_value(gpu_batch[2])

                for idx in range(nb_batch_store_gpu):

                    batch_idx += 1
                    Z = rng.uniform(low=-1, high=1, size=(batch_size, Z_dim)).astype('float32')
                    #Z = rng.normal(0, 1, size=(batch_size, Z_dim)).astype('float32')

                    g_cost, d_cost, k_tp, convergence_measure, gamma, loss_center = self.train_g_d_fn(idx, Z)
                    print('Batch %06d/%06d - Epoch %04d -- Loss G/D: %.4f/%.4f -- k: %.4f -- convergence: %.4f '
                          ' -- gamma: %04f -- loss center: %.4f ' %
                          (batch_idx, nb_batch_train, epoch, g_cost, d_cost, k_tp, convergence_measure,
                           gamma, loss_center))
                    # TODO: copy loss to some file
                    f = open('loss_began.txt', 'a')
                    f.write('%06d, %.4f, %.4f, %.4f, %.4f, %.4f\n' %
                            (batch_idx, g_cost, d_cost, k_tp, convergence_measure, gamma))
                    f.close()

                    #if batch_idx % 400 == 0:

                      #  self.save_img(Z, 0, batch_size, 'train2014')
                      #  self.save('model_wgan_01.pkl')
            if epoch % 3 == 0:
                self.save('model_began_2_epoch_%04d.pkl' % (epoch + 87))

    def denorm_img(self, img):
        # image pixel values are cast from [-1, 1] -> [0,255]
        img = ((img + 1.) / 2.) * 255.
        img = T.switch(T.ge(img, 255), 255, img)
        img = T.switch(T.le(img, 0), 0, img)
        return img

    def save(self, file_path):
        """ Save the model """
        f = open(file_path, 'wb')
        cPickle.dump((self.params.values(), self.hparams.values()), f)
        f.close()

    def generate(self, Z, batch_idx, batch_size, dataset):
        data = load_data(batch_idx, batch_size, dataset)
        self.gpu_dataset_Y.set_value(data[1])
        self.gpu_dataset_X.set_value(data[0])
        self.gpu_dataset_emb.set_value(data[2])
        imgs_g, imgs_d, d_gen_out = self.generate_image(0, Z)
        return imgs_g, imgs_d, d_gen_out, data[1]

    def save_img(self, Z, batch_idx, batch_size, dataset):
        """ Save an image """
        img = self.generate(Z, batch_idx, batch_size, dataset)
        #i_g = ((img[0][0].transpose((1, 2, 0)) + 1.) / 2.) * 255.
        #i_d = ((img[1][0].transpose((1, 2, 0)) + 1.) / 2.) * 255.
        #d_gen_out = ((img[2][0].transpose((1, 2, 0)) + 1.) / 2.) * 255.
        i_g = img[0][0].transpose((1, 2, 0))
        i_d = img[1][0].transpose((1, 2, 0))
        d_gen_out = img[2][0].transpose((1, 2, 0))
        #i2 = ((img[0][0][1].transpose((1, 2, 0)))) * 255.
        #i3 = ((img[0][0][3].transpose((1, 2, 0)))) * 255.
        Image.fromarray(i_g.astype('uint8')).show()
        Image.fromarray(i_d.astype('uint8')).show()
        Image.fromarray(d_gen_out.astype('uint8')).show()
        #Image.fromarray(i2.astype('uint8')).show()
        #Image.fromarray(i3.astype('uint8')).show()

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

    def show_imgs_dis(self, Z, batch_idx, batch_size, dataset):
        imgs = self.generate(Z, batch_idx, batch_size, dataset)
        for img in imgs[2]:
            #Image.fromarray((((img.transpose((1, 2, 0)) + 1. )
             #                / 2. ) * 255.).astype('uint8'), 'RGB').show()
            Image.fromarray(img.transpose((1, 2, 0)).astype('uint8'), 'RGB').show()


        # TODO: add image id's to some file to capture captions

    def show_imgs_gen(self, Z, batch_idx, batch_size, dataset):
        imgs = self.generate(Z, batch_idx, batch_size, dataset)
        for img in imgs[0]:
            #Image.fromarray((((img.transpose((1, 2, 0)) + 1. )
            #                 / 2. ) * 255.).astype('uint8'), 'RGB').show()
            Image.fromarray(img.transpose((1, 2, 0)).astype('uint8'), 'RGB').show()

    def show_imgs_true(self, Z, batch_idx, batch_size, dataset):
        imgs = self.generate(Z, batch_idx, batch_size, dataset)
        for img in imgs[1]:
            #Image.fromarray((((img.transpose((1, 2, 0)) + 1. )
            #                 / 2. ) * 255.).astype('uint8'), 'RGB').show()
            Image.fromarray(img.transpose((1, 2, 0)).astype('uint8'), 'RGB').show()


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
        """ Get Hyper-parameters """
        return {
            'batch_size':   64,                # size of minibatches
            'learning_rate':  0.00005,             # learning rate
            'Z_dim':        100,
            'nb_epochs':    500,
            'nb_batch_store_gpu': 10 ,
            "l2":           2.5e-5,  # l2 weight decay
            "b1":           0.5,  # momentum term of adam
            'dropout_p': 0.3,
            'dis_f': 128,
            'mb_kernel_dim': 5,
            'mb_nb_kernels': 100,
            'n_crit':   5,


        }

if __name__ == '__main__':
    network = BEGAN()
    network.build_model()
    #network.load('lib_models/model_began_2_epoch_0027.pkl')
    network.load('model_began_2_epoch_0087.pkl')
    network.train_model()