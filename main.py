import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a model')
    parser.add_argument('model', action="store", type=str, dest='model')
    args = parser.parse_args()

    if args.model == 'BEGAN':
        from lib_models.began_squ import *
        network = BEGAN()
        network.build_model()
        network.train_model()
        
    elif args.model == 'GAN':
        from lib_models.gan_hole_reconstruction import *
        network = GenerativeAdversarialNetwork()
        network.build_model()
        network.train_model()
        
    elif args.model == 'WGAN':
        from lib_models.wgan import *
        network = WGAN()
        network.build_model()
        network.train_model()
