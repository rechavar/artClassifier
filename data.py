import os
import sys
import json 
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import cv2
import tensorflow as tf
import tensorflow.io as io

label = {'fotografia': 0, 'Escultura':1, 'Mural':2, 'pintura':3, 'dibujo':4}

def prepareDataset(dataDir):
    if not os.path.exists('image_files'):
        os.makedirs('image_files')
    
    annDir = os.path.join(dataDir, 'ann')
    annList = [file for file in os.listdir(annDir) if os.path.isfile(os.path.join(annDir, file))]
    metadata = defaultdict(list)
    fn = lambda x: str(hash(x) % ((sys.maxsize + 1) * 2)) + '.PNG'
    for ann in annList:
        with open(os.path.join(annDir, ann),'r') as f:
            dataStore = json.loads(f.read())
        for i in dataStore['objects']:
            newFileName = fn(ann)

            try:
                metadata['imageLabel'].append(i['tags'][0]['name'])
            except:
                print(ann)
                break
            metadata['imageName'].append(newFileName)
            saveCropImage(os.path.join(annDir[:-4],'img',ann[:-5]), newFileName, i['points']['exterior'])
    metadata['split'] = splitDataset(len(metadata['imageLabel']))
    metadata = pd.DataFrame(metadata)
    metadata.to_csv('metadata.csv', index=False)
    metadata1 = []
    for _, values in label.items():
        metadata1.append(metadata.query("split == 'train' & label == " + str(values).iloc[0]))
    pd.DataFrame(metadata1).to_csv('metadata1.csv', index = False)


def saveCropImage(filePath, newName, exterior):
    img = cv2.imread(filePath)
    img2 = img[int(exterior[1][0]):int(exterior[0][0]), int(exterior[1][1]):int(exterior[1][0])]
    cv2.imwrite(os.path.join('image file',newName),img2)

def splitDataset(ds_len):
    test =  int(ds_len * 0.2)
    split = ['test'] * test
    train = int(ds_len * 0.7)
    split.extend(['train'] * train)
    val = ds_len - (test + train)
    split.extend(['val'] * val)
    random.shuffle(split)
    return split


def buildSources(metadata, dataDir, mode = 'train', excludeLabel = None):
    
    if excludeLabel is None:
        excludeLabel = set()
    elif isinstance(excludeLabel,(list,tuple)):
        excludeLabel = set(excludeLabel)
    
    df = metadata[metadata['split'] == mode]
    df['filepath'] = metadata['imageName'].apply(lambda x: os.path.join(dataDir.x))
    includeMask = metadata['label'].apply(lambda x: c not in excludeLabel)
    df = df[includeMask]

    sources = list(zip(df['filepath'],df['label']))
    return sources


def preprocessImage(image,pixels):
    return  (tf.image.resize(image, size=(pixels,pixels)))/255.0

def make_dataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None, pixels = 32, target = 1):
    """
    Returns an operation to iterate over the dataset specified in sources
    Args:
        sources (list): A list of (filepath, label_id) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_epochs (int): Number of epochs to repeat the dataset.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.
        shuffle_buffer_size (int): Number of elements from this dataset
            from which the new dataset will sample.
        pixels (int): Size of the image after resize 
    Returns:
        A tf.data.Dataset object. It will return a tuple images of shape
        [N, H, W, CH] and labels shape [N, 1].
    """
    def load(row):
        filepath = row['image']
        img = io.decode_jpeg(io.read_file(filepath))
        return img, row['label']
    
    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size*4  ##preguntar el porque de ese 4
    
    image,labels = zip(*sources)

    ds = tf.data.Dataset.from_tensor_slices({
        'image' : list(image),
        'label' : list(labels)
    })

    if training:
        ds = ds.shuffle(shuffle_buffer_size0)

    ds = map(load,num_parallel_calls = num_parallel_calls)
    ds = map(lambda x,y: (preprocessImage(x,pixels), y))
    
    if training:
        ds = ds.map(lambda x,y: augmentImage(x),y)
    
    ds = ds.map(lambda x, y: (x, tuple([y]*target) if target > 1 else y))
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds

def imshowThree(batch):

    labelBatch = batch[1].numpy()
    imageBatch = batch[0].numpy()
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i in range(3):
        img = imageBatch[i,...]
        axarr[i].imshow(img)
        axarr[i].set(xlabel='label = {}'.format(labelBatch[i]))
    
def augmentImage(img):
    return img
def drawResults(H,N,name, val = False):
    fig,axis = plt.subplots(2)
    fig.suptitle('Training loss and accuracy on Dataset')
    axis[0].plot(np.range(0,N),H.history["Loss"],label = "trainLoss")

    if val:
        axis[0].plot(np.arange(0,N),H.history["valLoss"],label = "valLoss")
    axs[0].set_xlabel("Epoch #")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="lower left")
    axs[1].plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

    if(val):
        axs[1].plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    axs[1].set_xlabel("Epoch #")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc="lower left")
    plt.savefig(name + ".png")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare dataset")

    parser.add_argument('--dir', '-d',
        help="directory with images"
    )
    
    args = parser.parse_args()
    prepare_dataset(args.dir)
    