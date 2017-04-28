# Most of the code is inspired from the Deep Learning tutorial at
# http://deeplearning.net, from https://github.com/ppaquette
# /ift-6266-project/blob/master/models/auto_encoder.py, especially
# the copying of multiple minibatches on the GPU instead of
# loading data minibatch by minibatch which resulted in a
# significant increase in performance (computational time - wise) and
# from the DCGAN github repo https://github.com/Newmu/dcgan_code.

# We try to implement a deep residual convolutional autoencoder
# like https://arxiv.org/pdf/1606.08921.pdf the implementation
# of residual blocks is the same as https://arxiv.org/pdf/1512.03385.pdf
from lib.ops import *
from lib.data_utils import *
from lib.inits import *
from lib.updates import *
from lib.initializations import *
from lib.theano_utils import *
import numpy as np
import theano, theano.tensor as T
import timeit
import collections
import cPickle

theano.config.exception_verbosity='high'
class AutoEncoder(object):
    """ Deep Residual Convolutional AutoEncoder Network """

    def save(self, file_path):
        """ Save the model """
        f = open(file_path, 'wb')
        cPickle.dump((self.params.values(), self.hparams.values()), f)
        f.close()

    def generate(self):
        data = load_data(0, 128, 'val2014')
        self.gpu_dataset_Y.set_value(data[1])
        self.gpu_dataset_X.set_value(data[0])
        imgs = self.generate_image(0)
        return imgs, data[1]

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
            self.params[keys[i]] = values[i]
        return params

    def build_model(self):
        """ Build the AutoEncoder """

        # load the hyperparameters
        self.hparams = self.get_hparams()
        hp = self.hparams

        X = T.tensor4(name='X', dtype='float32')
        Y = T.tensor4(name='Y', dtype='float32')
        index = T.iscalar('index')

        batch_size = hp['batch_size']
        l_rate = hp['l_rate']

        # parameters initialization
        self.params = collections.OrderedDict()
        self.params['W_conv1'] = He(hp['f1'], name='W_conv1', fan_in=np.prod(hp['f1'][1:]))
        self.params['b_conv1'] = constant((hp['f1'][0], ), name='b_conv1')
        self.params['W_conv2'] = He(hp['f2'], name='W_conv2', fan_in=np.prod(hp['f2'][1:]))
        self.params['b_conv2'] = constant((hp['f2'][0], ), name='b_conv2')

        self.params['W_conv3'] = He(hp['f3'], name='W_conv3', fan_in=np.prod(hp['f3'][1:]))
        self.params['b_conv3'] = constant((hp['f3'][0], ), name='b_conv3')
        self.params['W_conv4'] = He(hp['f4'], name='W_conv4', fan_in=np.prod(hp['f4'][1:]))
        self.params['b_conv4'] = constant((hp['f4'][0], ), name='b_conv4')

        self.params['W_conv5'] = He(hp['f5'], name='W_conv5', fan_in=np.prod(hp['f5'][1:]))
        self.params['b_conv5'] = constant((hp['f5'][0], ), name='b_conv5')
        self.params['W_conv6'] = He(hp['f6'], name='W_conv6', fan_in=np.prod(hp['f6'][1:]))
        self.params['b_conv6'] = constant((hp['f6'][0], ), name='b_conv6')

        self.params['W_conv7'] = He(hp['f7'], name='W_conv7', fan_in=np.prod(hp['f7'][1:]))
        self.params['b_conv7'] = constant((hp['f7'][0], ), name='b_conv7')
        self.params['W_conv8'] = He(hp['f8'], name='W_conv8', fan_in=np.prod(hp['f8'][1:]))
        self.params['b_conv8'] = constant((hp['f8'][0], ), name='b_conv8')

        self.params['W_conv9'] = He(hp['f9'], name='W_conv9', fan_in=np.prod(hp['f9'][1:]))
        self.params['b_conv9'] = constant((hp['f9'][0], ), name='b_conv9')
        self.params['W_conv10'] = He(hp['f10'], name='W_conv10', fan_in=np.prod(hp['f10'][1:]))
        self.params['b_conv10'] = constant((hp['f10'][0], ), name='b_conv10')

        self.params['W_conv11'] = He(hp['f11'], name='W_conv11', fan_in=np.prod(hp['f11'][1:]))
        self.params['b_conv11'] = constant((hp['f11'][0], ), name='b_conv11')
        self.params['W_conv12'] = He(hp['f12'], name='W_conv12', fan_in=np.prod(hp['f12'][1:]))
        self.params['b_conv12'] = constant((hp['f12'][0], ), name='b_conv12')

        self.params['W_conv13'] = He(hp['f13'], name='W_conv13', fan_in=np.prod(hp['f13'][1:]))
        self.params['b_conv13'] = constant((hp['f13'][0], ), name='b_conv13')
        self.params['W_conv14'] = He(hp['f14'], name='W_conv14', fan_in=np.prod(hp['f14'][1:]))
        self.params['b_conv14'] = constant((hp['f14'][0], ), name='b_conv14')

        self.params['W_conv15'] = He(hp['f15'], name='W_conv15', fan_in=np.prod(hp['f15'][1:]))
        self.params['b_conv15'] = constant((hp['f15'][0], ), name='b_conv15')
        self.params['W_conv16'] = He(hp['f16'], name='W_conv16', fan_in=np.prod(hp['f16'][1:]))
        self.params['b_conv16'] = constant((hp['f16'][0], ), name='b_conv16')

        self.params['W_deconv1'] = He(hp['f17'], name='W_deconv1', fan_in=np.prod(hp['f17'][1:]))
        self.params['b_deconv1'] = constant((hp['f17'][1], ), name='b_deconv1')
        self.params['W_deconv2'] = He(hp['f18'], name='W_deconv2', fan_in=np.prod(hp['f18'][1:]))
        self.params['b_deconv2'] = constant((hp['f18'][1], ), name='b_deconv2')

        self.params['W_deconv3'] = He(hp['f19'], name='W_deconv3', fan_in=np.prod(hp['f19'][1:]))
        self.params['b_deconv3'] = constant((hp['f19'][1], ), name='b_deconv3')
        self.params['W_deconv4'] = He(hp['f20'], name='W_deconv4', fan_in=np.prod(hp['f20'][1:]))
        self.params['b_deconv4'] = constant((hp['f20'][1], ), name='b_deconv4')

        self.params['W_deconv5'] = He(hp['f21'], name='W_deconv5', fan_in=np.prod(hp['f21'][1:]))
        self.params['b_deconv5'] = constant((hp['f21'][1], ), name='b_deconv5')
        self.params['W_deconv6'] = He(hp['f22'], name='W_deconv6', fan_in=np.prod(hp['f22'][1:]))
        self.params['b_deconv6'] = constant((hp['f22'][1], ), name='b_deconv6')

        self.params['W_deconv7'] = He(hp['f23'], name='W_deconv7', fan_in=np.prod(hp['f23'][1:]))
        self.params['b_deconv7'] = constant((hp['f23'][1], ), name='b_deconv7')
        self.params['W_deconv8'] = He(hp['f24'], name='W_deconv8', fan_in=np.prod(hp['f24'][1:]))
        self.params['b_deconv8'] = constant((hp['f24'][1], ), name='b_deconv8')

        self.params['W_deconv9'] = He(hp['f25'], name='W_deconv9', fan_in=np.prod(hp['f25'][1:]))
        self.params['b_deconv9'] = constant((hp['f25'][1], ), name='b_deconv9')
        self.params['W_deconv10'] = He(hp['f26'], name='W_deconv10', fan_in=np.prod(hp['f26'][1:]))
        self.params['b_deconv10'] = constant((hp['f26'][1], ), name='b_deconv10')

        self.params['W_deconv11'] = He(hp['f27'], name='W_deconv11', fan_in=np.prod(hp['f27'][1:]))
        self.params['b_deconv11'] = constant((hp['f27'][1], ), name='b_deconv11')
        self.params['W_deconv12'] = He(hp['f28'], name='W_deconv12', fan_in=np.prod(hp['f28'][1:]))
        self.params['b_deconv12'] = constant((hp['f28'][1], ), name='b_deconv12')

        self.params['W_deconv13'] = He(hp['f29'], name='W_deconv13', fan_in=np.prod(hp['f29'][1:]))
        self.params['b_deconv13'] = constant((hp['f29'][1], ), name='b_deconv13')
        self.params['W_deconv14'] = He(hp['f30'], name='W_deconv14', fan_in=np.prod(hp['f30'][1:]))
        self.params['b_deconv14'] = constant((hp['f30'][1], ), name='b_deconv14')

        self.params['W_deconv15'] = He(hp['f31'], name='W_deconv15', fan_in=np.prod(hp['f31'][1:]))
        self.params['b_deconv15'] = constant((hp['f31'][1], ), name='b_deconv15')
        self.params['W_deconv16'] = He(hp['f32'], name='W_deconv16', fan_in=np.prod(hp['f32'][1:]))
        self.params['b_deconv16'] = constant((hp['f32'][1], ), name='b_deconv16')
        p = self.params

        print('Building the model...')

        def encode(X):
            # 4x Conv2D [3x3] + 4x Deconv2D [3x3] with residual
            # connections between corresponding Conv2D <=> Deconv2D
            input = X

            conv1 = relu(Res_conv(input, hp['f1'], p['W_conv1']) +
                         p['b_conv1'].dimshuffle('x', 0, 'x', 'x'))
            conv2 = relu(Res_conv(conv1, hp['f2'], p['W_conv2']) +
                         p['b_conv2'].dimshuffle('x', 0, 'x', 'x'))
            conv3 = relu(Res_conv(conv2, hp['f3'], p['W_conv3']) +
                         p['b_conv3'].dimshuffle('x', 0, 'x', 'x'))

            conv4 = relu(Res_conv(conv3, hp['f4'], p['W_conv4']) +
                         p['b_conv4'].dimshuffle('x', 0, 'x', 'x'))
            conv5 = relu(Res_conv(conv4, hp['f5'], p['W_conv5']) +
                         p['b_conv5'].dimshuffle('x', 0, 'x', 'x'))
            conv6 = relu(Res_conv(conv5, hp['f6'], p['W_conv6']) +
                         p['b_conv6'].dimshuffle('x', 0, 'x', 'x'))

            conv7 = relu(Res_conv(conv6, hp['f7'], p['W_conv7']) +
                         p['b_conv7'].dimshuffle('x', 0, 'x', 'x'))
            conv8 = relu(Res_conv(conv7, hp['f8'], p['W_conv8']) +
                         p['b_conv8'].dimshuffle('x', 0, 'x', 'x'))
            conv9 = relu(Res_conv(conv8, hp['f9'], p['W_conv9']) +
                         p['b_conv9'].dimshuffle('x', 0, 'x', 'x'))

            conv10 = relu(Res_conv(conv9, hp['f10'], p['W_conv10']) +
                          p['b_conv10'].dimshuffle('x', 0, 'x', 'x'))
            conv11 = relu(Res_conv(conv10, hp['f11'], p['W_conv11']) +
                          p['b_conv11'].dimshuffle('x', 0, 'x', 'x'))
            conv12 = relu(Res_conv(conv11, hp['f12'], p['W_conv12']) +
                          p['b_conv12'].dimshuffle('x', 0, 'x', 'x'))

            conv13 = relu(Res_conv(conv12, hp['f13'], p['W_conv13']) +
                          p['b_conv13'].dimshuffle('x', 0, 'x', 'x'))
            conv14 = relu(Res_conv(conv13, hp['f14'], p['W_conv14']) +
                          p['b_conv14'].dimshuffle('x', 0, 'x', 'x'))


            conv15 = relu(Res_conv(conv14, hp['f15'], p['W_conv15']) +
                          p['b_conv15'].dimshuffle('x', 0, 'x', 'x'))


            conv16 = relu(Res_conv(conv15, hp['f16'], p['W_conv16'])  +
                          p['b_conv16'].dimshuffle('x', 0, 'x', 'x'))


            #encoder = pool2D(relu(Res_conv(conv15, hp['f16'], p['W_conv16']) + p['b_conv16'].dimshuffle('x', 0, 'x', 'x')))



            deconv1 = relu(Res_deconv(conv16, hp['f17'], p['W_deconv1'], hp['i5'])     #4
                           + p['b_deconv1'].dimshuffle('x', 0, 'x', 'x'))
            deconv1 = relu(deconv1 + conv15)
            deconv2 = relu(Res_deconv(deconv1, hp['f18'], p['W_deconv2'], hp['i6'])   #8
                                      + p['b_deconv2'].dimshuffle('x', 0, 'x', 'x'))
            deconv2 = relu(deconv2 + conv14)
            deconv3 = relu(Res_deconv(deconv2, hp['f19'], p['W_deconv3'], hp['i7'])
                                      + p['b_deconv3'].dimshuffle('x', 0, 'x', 'x'))
            deconv3 = relu(deconv3 + conv13)
            deconv4 = relu(Res_deconv(deconv3, hp['f20'], p['W_deconv4'], hp['i8'])
                                      + p['b_deconv4'].dimshuffle('x', 0, 'x', 'x'))
            deconv4 = relu(deconv4 + conv12)
            deconv5 = relu(Res_deconv(deconv4, hp['f21'], p['W_deconv5'], hp['i9'])
                           + p['b_deconv5'].dimshuffle('x', 0, 'x', 'x'))
            deconv5 = relu(deconv5 + conv11)
            deconv6 = relu(Res_deconv(deconv5, hp['f22'], p['W_deconv6'], hp['i10'])
                                      + p['b_deconv6'].dimshuffle('x', 0, 'x', 'x'))
            deconv6 = relu(deconv6 + conv10)
            deconv7 = relu(Res_deconv(deconv6, hp['f23'], p['W_deconv7'], hp['i11'])
                                      + p['b_deconv7'].dimshuffle('x', 0, 'x', 'x'))
            deconv7 = relu(deconv7 + conv9)

            deconv8 = relu(Res_deconv(deconv7, hp['f24'], p['W_deconv8'], hp['i12'])
                                      + p['b_deconv8'].dimshuffle('x', 0, 'x', 'x'))
            deconv8 = relu(deconv8 + conv8)

            deconv9 = relu(Res_deconv(deconv8, hp['f25'], p['W_deconv9'], hp['i13'])
                           + p['b_deconv9'].dimshuffle('x', 0, 'x', 'x'))
            deconv9 = relu(deconv9 + conv7)
            deconv10 = relu(Res_deconv(deconv9, hp['f26'], p['W_deconv10'], hp['i14'])
                                      + p['b_deconv10'].dimshuffle('x', 0, 'x', 'x'))
            deconv10 = relu(deconv10 + conv6)
            deconv11 = relu(Res_deconv(deconv10, hp['f27'], p['W_deconv11'], hp['i15'])
                                      + p['b_deconv11'].dimshuffle('x', 0, 'x', 'x'))
            deconv11 = relu(deconv11 + conv5)
            deconv12 = relu(Res_deconv(deconv11, hp['f28'], p['W_deconv12'], hp['i16'])
                                      + p['b_deconv12'].dimshuffle('x', 0, 'x', 'x'))

            deconv12 = relu(deconv12 + conv4)
            deconv13 = relu(Res_deconv(deconv12, hp['f29'], p['W_deconv13'], hp['i17'])
                                      + p['b_deconv13'].dimshuffle('x', 0, 'x', 'x'))
            deconv13 = relu(deconv13 + conv3)
            deconv14 = relu(Res_deconv(deconv13, hp['f30'], p['W_deconv14'], hp['i18'])
                                      + p['b_deconv14'].dimshuffle('x', 0, 'x', 'x'))
            deconv14 = relu(deconv14 + conv2)
            deconv15 = relu(Res_deconv(deconv14, hp['f31'], p['W_deconv15'], hp['i19'])
                                      + p['b_deconv15'].dimshuffle('x', 0, 'x', 'x'))
            deconv15 = relu(deconv15 + conv1)
            deconv16 = relu(Res_deconv(deconv15, hp['f32'], p['W_deconv16'], hp['i20'])
                                      + p['b_deconv16'].dimshuffle('x', 0, 'x', 'x'))
            deconv16 = relu(deconv16 + input)
            return deconv16

        images = encode(X)

        # Combine the real "contour" with the generated center
        #mask = T.zeros((batch_size, 3, 64, 64), dtype='float32')
        #mask = T.set_subtensor(mask[:, :, 16:48, 16:48], 1.)

        #images = mask * images + (1 - mask) * Y
        loss = 1000. * T.mean((images - Y) ** 2)

        print('Compiling model...')
        # ----------------------------
        # check this when using 32x32
        self.gpu_dataset_X = shared0s((10 * batch_size, 3, 64, 64), 'float32')
        self.gpu_dataset_Y = shared0s((10 * batch_size, 3, 64, 64), 'float32')

        updater = Adam(lr=l_rate, b1=0.5, regularizer=Regularizer(l2=1e-5))
        updates = updater(p.values(), loss)

        self.generate_image = theano.function(
            [index],
            [images],
            givens={
                X: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size],
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size]
            },
            on_unused_input='warn'
        )

        self.test_fn = theano.function(
            [index],
            loss,
            givens={
                X: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size],
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.train_fn = theano.function(
            [index],
            [loss],
            updates=updates,
            givens={
                X: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size],
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size]
            }
        )

    def train_model(self):
        """ Train the model """
        print('Training...')
        self.hparams = self.get_hparams()
        hp = self.hparams
        batch_size = hp['batch_size']
        n_epochs = hp['n_epochs']

        n_train_batches = 82611 // batch_size
        n_test_batches = 40438 // batch_size


        n_valid_batches = n_test_batches

        n_train_gpu_batch = n_train_batches // 10
        n_test_gpu_batch = n_test_batches // 10
        n_valid_gpu_batch = n_valid_batches // 10

        patience = hp['patience']
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(n_train_batches, patience // 2)
        best_validation_loss = np.inf
        start_time = timeit.default_timer()

        COMPUTE_VALIDATION = False
        DONE_LOOPING = False
        epoch = 0
        while epoch < n_epochs and not DONE_LOOPING:
            epoch += 1
            masterbatch_idx = 0
            minibatch_avg_loss = []

            # loop through the GPU_batches
            for gpu_batch in iterate_minibatches(batch_size * 10, 'train2014'):

                self.gpu_dataset_Y.set_value(gpu_batch[1])
                self.gpu_dataset_X.set_value(gpu_batch[0])


                for idx in range(10):
                    loss = self.train_fn(idx)
                    #minibatch_avg_loss.append(self.train_fn(idx))
                    minibatch_avg_loss.append(loss)
                    masterbatch_idx += 1

                    # Compute the number of iterations
                    n_iter = (epoch - 1) * n_train_batches + masterbatch_idx

            # Compute training loss
            train_avg_loss = np.mean(minibatch_avg_loss)
            print('epoch %i, train error: %i' % (epoch, train_avg_loss))

            # Compute validation loss
            print('Computing validation loss for epoch %i' % epoch)
            valid_avg_loss = []
            for gpu_batch in iterate_minibatches(batch_size * 10, 'val2014'):
                self.gpu_dataset_X.set_value(gpu_batch[0])
                self.gpu_dataset_Y.set_value(gpu_batch[1])
                for idx in range(10):
                    valid_avg_loss.append(self.test_fn(idx))

            validation_loss = np.mean(valid_avg_loss)
            print('epoch %i, validation loss: %i' % (epoch, validation_loss))

            # Check if we beated last validation loss
            if validation_loss < best_validation_loss:

                # improve patience if loss improvement is good enough
                if validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, n_iter * patience_increase)
                best_validation_loss = validation_loss

                # Save the model
                print('Saving...')
                self.save('model_drcae_64x64.pkl')
                # -------------

            if patience <= n_iter:
                DONE_LOOPING = True
                break


        end_time = timeit.default_timer()
        print('Optimization complete')
        # Test the model
        # -------------




    def get_hparams(self):
        """ Get Hyperparameters """
        return {
            'batch_size':   32,                # size of minibatches
            'l_rate':       0.0002,             # learning rate
            'patience':     10000,              # patience for early stopping
            'n_epochs':     500,                # number of epochs

            'f1':           (32, 3, 5, 5),      # convolutions
            'f2':           (32, 32, 5, 5),
            'f3':           (32, 32, 5, 5),
            'f4':           (32, 32, 5, 5),
            'f5':           (32, 32, 5, 5),
            'f6':           (32, 32, 5, 5),
            'f7':           (32, 32, 5, 5),
            'f8':           (32, 32, 5, 5),
            'f9':           (32, 32, 5, 5),
            'f10':          (32, 32, 5, 5),
            'f11':          (32, 32, 5, 5),
            'f12':          (32, 32, 5, 5),
            'f13':          (32, 32, 5, 5),
            'f14':          (32, 32, 5, 5),
            'f15':          (32, 32, 5, 5),

            'f16':           (100, 32, 4, 4),      # encoder -- 100 units

            'f17':           (100, 32, 4, 4),        # deconvolutions
            'f18':           (32, 32, 5, 5),
            'f19':           (32, 32, 5, 5),
            'f20':           (32, 32, 5, 5),
            'f21':          (32, 32, 5, 5),
            'f22':          (32, 32, 5, 5),
            'f23':          (32, 32, 5, 5),
            'f24':          (32, 32, 5, 5),
            'f25':          (32, 32, 5, 5),
            'f26':          (32, 32, 5, 5),
            'f27':          (32, 32, 5, 5),
            'f28':          (32, 32, 5, 5),
            'f29':          (32, 32, 5, 5),
            'f30':          (32, 32, 5, 5),
            'f31':          (32, 32, 5, 5),
            'f32':          (32, 3, 5, 5),

            'i5':           (32, 32, 4, 4),         # output shapes deconv
            'i6':           (32, 32, 8, 8),
            'i7':           (32, 32, 12, 12),
            'i8':           (32, 32, 16, 16),
            'i9':           (32, 32, 20, 20),
            'i10':          (32, 32, 24, 24),
            'i11':          (32, 32, 28, 28),
            'i12':          (32, 32, 32, 32),
            'i13':          (32, 32, 36, 36),
            'i14':          (32, 32, 40, 40),
            'i15':          (32, 32, 44, 44),
            'i16':          (32, 32, 48, 48),
            'i17':          (32, 32, 52, 52),
            'i18':          (32, 32, 56, 56),
            'i19':          (32, 32, 60, 60),
            'i20':          (32, 3, 64, 64)
        }

if __name__ == '__main__':
    network = AutoEncoder()
    network.build_model()
    #network.load('model_drcae_64x64.pkl')
    network.train_model()
    #img = network.generate()
    #Image.fromarray(img[0][0][4].transpose((1,2,0)).astype('uint8')).show()
    #Image.fromarray(img[1][4].transpose((1, 2, 0)).astype('uint8')).show()
