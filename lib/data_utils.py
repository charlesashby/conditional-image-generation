import os
import numpy as np
import glob
import Image
import nltk
from nltk.corpus import stopwords
import cPickle
import string
import scipy
import PIL
from PIL import Image
import scipy.ndimage

# -------------
# Load encoder from skipthoughts:
#
# from skip_thoughts import skipthoughts
# model = skipthoughts.load_model()
# encoder = skipthoughts.Encoder(model)
# X = ['Ten birds standing in a tree.', 'Four man playing baseball.',
#  'A red sauce in a black bowl on a table.', 'A black bus.',
#  'A bottle of wine with a red sticker next to a glass of wine.',
#  'A brown bear with a red nose.']

# keys = ['COCO_val2014_000000000143', 'COCO_val2014_000000000192', 'COCO_val2014_000000000196',
#       'COCO_val2014_000000000257', 'COCO_val2014_000000000283', 'COCO_val2014_000000000285']

# vectors = encoder.encode(X)


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def load_data(batch_idx, batch_size, split, key=None, new_embedding=None,
              mscoco="/home/ashbylepoc/PycharmProjects/DeepLearning/inpainting",
              caption_path="dict_key_imgID_value_caps_train_and_valid.pkl",
              output_mode='64x64'):
    """ Load the dataset """

    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    if split == 'train2014' or split == 'train':
        f = open(os.path.join(mscoco, 'embedding_dict_train.pkl'), 'rb')
        dict_keys = cPickle.load(f)
        f.close()
    elif split == 'val2014':
        f = open(os.path.join(mscoco, 'embedding_dict_val.pkl'), 'rb')
        dict_keys = cPickle.load(f)
        f.close()
    else:
        dict_keys = load_test_embedding(key, new_embedding)

    for i, img_path in enumerate(batch_imgs):

        # ---------------------
        # Captions do embedding
        # ---------------------
        key = find_between(img_path, split + '/', '.jpg')

        e = np.array([dict_keys[key]])

        img = Image.open(img_path)

        img_array = np.array(img)
        img_array = np.array([img_array])



        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[1] / 2.)),
                  int(np.floor(img_array.shape[2] / 2.)))
        img_array = img_array.transpose((0, 3, 1, 2))

        rand = np.random.uniform()
        if rand > 0.50:
            for c in range(len(img_array[0])):
                img_array[0][c] = scipy.ndimage.interpolation.rotate(img_array[0][c], angle=90)


        input = np.copy(img_array)

        mean_color = np.mean(img_array.flatten())
        #wrong_img = np.copy(input)
        if split == 'train2014' or split == 'val2014' or split == 'test':
            input[:, :, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16] = mean_color
        else:
            input[:, :, center[0] - 32:center[0] + 32, center[1] - 32:center[1] + 32] = mean_color


        #for c in range(len(wrong_img[0])):
        #    wrong_img[0][c] = scipy.ndimage.interpolation.rotate(wrong_img[0][c], angle=90)
        #wrong_cropped = np.copy(wrong_img)

        #wrong_cropped[:, :, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16] = 0

        if i == 0:
            train_set_x = input
            train_set_y = img_array
            train_set_emb = e
            #train_set_wrong_img = wrong_img
            #train_set_wrong_cropped = wrong_cropped
        else:
            train_set_x = np.concatenate((train_set_x, input), axis=0)
            train_set_y = np.concatenate((train_set_y, img_array), axis=0)
            train_set_emb = np.concatenate((train_set_emb, e), axis=0)
            #train_set_wrong_img = np.concatenate((train_set_wrong_img, wrong_img), axis=0)
            #train_set_wrong_cropped = np.concatenate((train_set_wrong_cropped, wrong_cropped), axis=0)
            # Image.fromarray(img_array).show()
            # Image.fromarray(input).show()
            # Image.fromarray(target).show()
            # print i, caption_dict[cap_id]

    #train_set_x = train_set_x.transpose((0, 3, 1, 2)) / 255.
    #train_set_y = train_set_y.transpose((0, 3, 1, 2)) / 255.
    #train_set_wrong_img = train_set_wrong_img.transpose(((0,3,1,2))) / 255.
    #train_set_wrong_cropped = train_set_wrong_cropped.transpose(((0,3,1,2))) / 255.
    train_set_x = np.asarray((train_set_x / 255. ) * 2. - 1., dtype='float32')
    train_set_y = np.asarray((train_set_y / 255. ) * 2. - 1., dtype='float32')
    #train_set_wrong_img = np.asarray(train_set_wrong_img * 2. - 1., dtype='float32')
    #train_set_wrong_cropped = np.asarray(train_set_wrong_cropped * 2. - 1., dtype='float32')
    #train_set_x = np.asarray(train_set_x , dtype='float32')
    #train_set_y = np.asarray(train_set_y , dtype='float32')
    #return (train_set_x, train_set_y, train_set_emb, train_set_wrong_cropped, train_set_wrong_img)
    return (train_set_x, train_set_y, train_set_emb)

def iterate_minibatches(batch_size, split, output_mode='64x64', l = 82611):
    """ Create an iterator for GPU batches """
    #if split == 'train2014':
    #    l = 82611
        #l = 25
    #else:
    #    l = 40438

    for start_idx in range(0, l // batch_size):

        #inputs, targets, embedding, wrong_cropped, wrong_img\
        #    = load_data(start_idx, batch_size, split, output_mode=output_mode)
        inputs, targets, embedding\
            = load_data(start_idx, batch_size, split, output_mode=output_mode)
        #yield np.asarray(inputs, dtype='float32'), np.asarray(targets, dtype='float32')
        #yield inputs, targets, embedding, wrong_cropped, wrong_img
        yield inputs, targets, embedding

def create_emb_dict(split, encoder,
              mscoco="/home/ashbylepoc/PycharmProjects/DeepLearning/inpainting",
              caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    """ Create a dictionary with the captions embedded """

    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")

    f = open(os.path.join(mscoco, caption_path))
    dict_keys = cPickle.load(f)
    f.close()
    emb_dict = {}
    temp_s = []             # temporary emplacement for sentences
    temp_k = []             # temporary emplacement for keys
    for i, img_path in enumerate(imgs):
        key = find_between(img_path, split + '/', '.jpg')
        sentence = dict_keys[key][0]
        temp_k.append(key)
        temp_s.append(sentence)
        #embedding = encoder.encode([sentence])
        #emb_dict[key] = embedding
    assert len(temp_k) == len(temp_s)
    print('Encoding...')
    embeddings = encoder.encode(temp_s)
    print('Loading in the dict...')
    for i in range(len(temp_k)):
        emb_dict[temp_k[i]] = embeddings[i]
    return emb_dict


def resize_image(split, save_dir = "/home/ashbylepoc/PycharmProjects/DeepLearning/inpainting/train",
                 mscoco="/home/ashbylepoc/PycharmProjects/DeepLearning/inpainting",
                 ):
    basewidth = 128
    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")

    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        key = find_between(img_path, split + '/', '.jpg')
        img.save(save_dir  + '/' + key + '.jpg')
        if i % 100 == 0:
            print('saved', key, 'to training/valid directory -- i:', i)



def change_captions(split, encoder,
              mscoco="/home/ashbylepoc/PycharmProjects/DeepLearning/inpainting",
              caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    """ Create a dictionary with the captions embedded """

    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")

    f = open(os.path.join(mscoco, caption_path))
    dict_keys = cPickle.load(f)
    f.close()
    emb_dict = {}
    temp_s = []             # temporary emplacement for sentences
    temp_k = []             # temporary emplacement for keys
    for i, img_path in enumerate(imgs):
        key = find_between(img_path, split + '/', '.jpg')
        sentence = dict_keys[key][0]
        temp_k.append(key)
        temp_s.append(sentence)





def load_test_embedding(key=None, new_embedding=None):
    f = open('/home/ashbylepoc/PycharmProjects/DeepLearning/inpainting/test_embedding.pkl', 'rb')
    dict_keys = cPickle.load(f)
    f.close()

    if key != None and new_embedding != None:
        for k in range(len(key)):
            dict_keys[key[k]] = new_embedding[k]
    return dict_keys
























