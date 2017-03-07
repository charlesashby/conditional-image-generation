# Most of the code is inspired from the Deep Learning tutorial at
# http://deeplearning.net and from https://github.com/ppaquette
# /ift-6266-project/blob/master/models/auto_encoder.py, especially
# the copying of multiple minibatches on the GPU instead of
# loading data minibatch by minibatch which resulted in a
# significant increase in performance (computational time - wise).
from lib.Layers import *
from lib.DataManagement import *
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
    """ Convolutional AutoEncoder Network """

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
        return imgs

    def load(self, file_path):
        """ Load model """
        f = open(file_path, 'rb')
        params = cPickle.load(f)
        f.close()
        params = params[0]

        self.params['W_conv1'].set_value(params[0].get_value())
        self.params['b_conv1'].set_value(params[1].get_value())
        self.params['W_conv2'].set_value(params[2].get_value())
        self.params['b_conv2'].set_value(params[3].get_value())
        self.params['W_hidden'].set_value(params[4].get_value())
        self.params['b_hidden'].set_value(params[5].get_value())
        self.params['W_visible'].set_value(params[6].get_value())
        self.params['b_visible'].set_value(params[7].get_value())
        self.params['W_deconv1'].set_value(params[8].get_value())
        self.params['b_deconv1'].set_value(params[9].get_value())
        self.params['W_deconv2'].set_value(params[10].get_value())
        self.params['b_deconv2'].set_value(params[11].get_value())
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
        self.params['W_hidden'] = He((hp['n_input'],hp['n_hidden']), name='W_hidden',
                                     fan_in=hp['n_hidden'])
        self.params['b_hidden'] = constant((hp['n_hidden'], ), name='b_hidden')
        self.params['W_visible'] = He((hp['n_hidden'],hp['n_visible']), name='W_visible',
                                     fan_in=hp['n_visible'])
        self.params['b_visible'] = constant((hp['n_visible'], ), name='b_visible')
        self.params['W_deconv1'] = He(hp['f3'], name='W_deconv1', fan_in=np.prod(hp['f3'][1:]))
        self.params['b_deconv1'] = constant((hp['f3'][1], ), name='b_deconv1')
        self.params['W_deconv2'] = He(hp['f4'], name='W_deconv2', fan_in=np.prod(hp['f4'][1:]))
        self.params['b_deconv2'] = constant((hp['f4'][1], ), name='b_deconv2')
        p = self.params

        print('Building the model...')

        def AutoEncoder(X):
            # Convolutional autoencoder with dense layer encoder + batchnorm
            input = X

            #conv1 = Conv2D(input, hp['f1'], hp['i1'], hp['b1'],
            #                hp['s1'], W=p['W_conv1'], b=p['b_conv1'])
            conv1 = relu(batchnorm(conv(input, hp['f1'], p['W_conv1'], hp['b1'], hp['s1']))
                         + p['b_conv1'].dimshuffle('x', 0, 'x', 'x'))
            conv1 = pool2D(conv1, hp['p1'])                                             # output of shape (128,64,30,30)
            #conv2 = Conv2D(conv1, hp['f2'], hp['i2'], hp['b2'],
            #                hp['s2'], W=p['W_conv2'], b=p['b_conv2'])
            conv2 = relu(batchnorm(conv(conv1, hp['f2'], p['W_conv2'], hp['b2'], hp['s2']))
                         + p['b_conv2'].dimshuffle('x', 0, 'x', 'x'))
            conv2 = pool2D(conv2, hp['p2'])                                             # output of shape (128,64,13,13)
            dense_layer = MLP(conv2, flatten=True, W_hidden=p['W_hidden'],
                              b_hidden=p['b_hidden'], W_visible=p['W_visible'],
                              b_visible=p['b_visible'])
            #deconv1 = TransposedConv2D(dense_layer, hp['f3'], hp['i3'], hp['s3'],
            #                  border_mode=hp['b3'], reshape=conv2.shape, W=p['W_deconv1'],
            #                  b=p['b_deconv1'])

            deconv1 = relu(batchnorm(deconv(dense_layer.reshape(conv2.shape),
                        hp['f3'], hp['i3'], p['W_deconv1'], hp['s3'],
                        hp['b3'])) + p['b_deconv1'].dimshuffle('x', 0, 'x', 'x'))
            deconv1 = Unpool2D(deconv1, hp['p3'])                                       # output of shape (128,64,30,30)
            #deconv2 = TransposedConv2D(deconv1, hp['f4'], hp['i4'], hp['s4'],
            #                  border_mode=hp['b4'], W=p['W_deconv2'], b=p['b_deconv2'])

            deconv1 = relu(deconv1 + conv1)
            deconv2 = relu(deconv(deconv1,
                        hp['f4'], hp['i4'], p['W_deconv2'], hp['s4'],
                        hp['b4']) + p['b_deconv2'].dimshuffle('x', 0, 'x', 'x'))
            deconv2 = Unpool2D(deconv2, hp['p4'])                                        # output of shape (128,3,64,64)
            return relu(deconv2 + input)

        images = AutoEncoder(X)
        # Combine the real "contour" with the generated center
        mask = T.zeros((batch_size, 3, 64, 64), dtype='float32')
        mask = T.set_subtensor(mask[:, :, 16:48, 16:48], 1.)

        images = mask * images + (1 - mask) * Y
        loss = 1000. * T.mean((images - Y) ** 2)

        print('Compiling model...')

        self.gpu_dataset_X = shared0s((10 * batch_size, 3, 64, 64), 'float32')
        self.gpu_dataset_Y = shared0s((10 * batch_size, 3, 64, 64), 'float32')

        updater = Adam(lr=l_rate, b1=0.5, regularizer=Regularizer(l2=1e-5))
        updates = updater(p.values(), loss)

        self.generate_image = theano.function(
            [index],
            [images, Y],
            givens={
                X: self.gpu_dataset_X[index * batch_size: (index + 1) * batch_size],
                Y: self.gpu_dataset_Y[index * batch_size: (index + 1) * batch_size]
            }
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
            [loss, images],
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
                    loss, img = self.train_fn(idx)
                    #minibatch_avg_loss.append(self.train_fn(idx))
                    minibatch_avg_loss.append(loss)
                    masterbatch_idx += 1
                    if loss == 0:
                        print img
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
                self.save('model_rcae_bn.pkl')
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
            'batch_size':   128,                # size of minibatches
            'l_rate':       0.0002,             # learning rate
            'patience':     10000,              # patience for early stopping
            'n_epochs':     500,                # number of epochs
            'i1':           (128, 3, 64, 64),   # input shape
            'f1':           (64, 3, 5, 5),      # filter_shape 1st convolution
            'b1':           (0, 0),             # border_shape 1st convolution
            's1':           (1, 1),             # subsample 1st convolution
            'p1':           (2, 2),             # pool_size 1st pooling
            'i2':           (128, 64, 30, 30),  # input shape 2nd convolution
            'f2':           (64, 64, 5, 5),     # filter_shape 2nd convolution
            'b2':           (0, 0),             # border_shape 2nd convolution
            's2':           (1, 1),             # subsample 2nd convolution
            'p2':           (2, 2),             # pool_size 2nd pooling
            'n_input':      10816,
            'n_hidden':     1024,                # hidden units dense layer
            'n_visible':    10816,              # visible units dense layer
            'i3':           (128, 64, 15, 15),  # input shape 1st deconv
            'f3':           (64, 64, 5, 5),     # filter_shape 1st deconv
            's3':           (1, 1),             # subsample 1st deconv
            'b3':           (1, 1),             # border_mode 1st deconv
            'p3':           (2, 2),             # pool_size 1st unpool layer
            'i4':           (128, 3, 32, 32),  # input shape 2nd deconv
            'f4':           (64, 3, 5, 5),      # filter_shape 2nd deconv
            's4':           (1, 1),             # subsample 2nd deconv
            'b4':           (1, 1),             # padding 2nd deconv
            'p4':           (2, 2)              # pool_size 2nd unpool layer
        }

if __name__ == '__main__':
    network = AutoEncoder()
    network.build_model()
    network.train_model()
    """
    from lib_models.rcae_bn import *
    network = AutoEncoder()
    network.build_model()
    network.load('lib_models/model_rcae_bn.pkl')
    t = network.generate()
    Image.fromarray(t[0][4].transpose((1, 2, 0)).astype('uint8'), 'RGB').show()
    Image.fromarray(t[0][5].transpose((1, 2, 0)).astype('uint8'), 'RGB').show()
    Image.fromarray(t[0][6].transpose((1, 2, 0)).astype('uint8'), 'RGB').show()
    Image.fromarray(t[0][7].transpose((1, 2, 0)).astype('uint8'), 'RGB').show()
    """

