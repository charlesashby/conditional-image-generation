import os
import numpy as np
import glob
import Image
import theano

def load_data(batch_idx, batch_size, split,
              mscoco="/home/ashbylepoc/PycharmProjects/DeepLearning/inpainting",
              rng=np.random.RandomState(1),
              caption_path="dict_key_imgID_value_caps_train_and_valid.pkl",
              output_mode='64x64'):
    """ Load the dataset """

    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    for i, img_path in enumerate(batch_imgs):

        img = Image.open(img_path)
        img_array = np.array(img)
        img_array = np.array([img_array])

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[1] / 2.)),
                  int(np.floor(img_array.shape[2] / 2.)))
        input = np.copy(img_array)
        input[:, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
        if output_mode == '32x32':
            target = img_array[:, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
        else:
            target = img_array[:,:,:,:]
        if i == 0:
            train_set_x = input
            train_set_y = target
        else:
            train_set_x = np.concatenate((train_set_x, input), axis=0)
            train_set_y = np.concatenate((train_set_y, target), axis=0)

            # Image.fromarray(img_array).show()
            # Image.fromarray(input).show()
            # Image.fromarray(target).show()
            # print i, caption_dict[cap_id]

    return (train_set_x.transpose((0, 3, 1, 2)), train_set_y.transpose((0, 3, 1, 2)))


def iterate_minibatches(batch_size, split, output_mode='64x64'):
    """ Create an iterator for GPU batches """
    if split == 'train2014':
        l = 82611
    else:
        l = 40438

    for start_idx in range(0, l // batch_size):
        if split == 'test':
            start_idx += l
        inputs, targets = load_data(start_idx, batch_size, split, output_mode=output_mode)
        yield np.asarray(inputs, dtype='float32'), np.asarray(targets, dtype='float32')

