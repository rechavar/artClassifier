import os
import sys
import json 
import random
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict
import numpy as np
import pandas as pd

labels = ['Escultura', 'Mural', 'fotografia', 'pintura', 'dibujo'] 

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

            saveCropImage(os.path.join(dataDir, os.path.join(annDir[:-4], ann)[:-5]), newFileName, i['points']['exterior'])

    metadata['split'] = splitDataset(len(metadata['imageLabel']))
    pd.DataFrame(metadata).to_csv('metadata.csv', index=False)
    metadata = pd.DataFrame(metadata)
    metadata1 = []
    for value in labels:
        metadata1.append(metadata.query("split == 'train' & imageLabel == " +"'"+ str(value)+"'").iloc[0]) 

    pd.DataFrame(metadata1).to_csv('metadata1.csv', index=False)

def saveCropImage(filePath, newName, exterior):
    img = cv2.imread(filePath)
    img2 = img[int(exterior[0][1]):int(exterior[1][1]), int(exterior[0][0]):int(exterior[1][0])]
    cv2.imwrite(os.path.join('image_files', newName), img2)

def splitDataset(ds_len):
    test =  int(ds_len * 0.2)
    split = ['test'] * test
    train = int(ds_len * 0.7)
    split.extend(['train'] * train)
    val = ds_len - (test + train)
    split.extend(['val'] * val)
    random.shuffle(split)
    return split

def buildSources(metadata, dataDir, mode='train', excludeLabels=None):
    
    if exclude_labels is None:
        exclude_labels = set()
    if isinstance(exclude_labels, (list, tuple)):
        exclude_labels = set(exclude_labels)

    df = metadata.copy()
    df = df[df['split'] == mode]
    df['filepath'] = df['image_name'].apply(lambda x: os.path.join(dataDir, x))
    include_mask = df['label'].apply(lambda x: x not in exclude_labels)
    df = df[include_mask]

    sources = list(zip(df['filepath'], df['label']))
    return sources

def preprocessImage(image, pixels):
    image = tf.image.resize(image, size=(pixels, pixels))
    image = image / 255.0
    return image

def makeDataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None, pixels = 32):
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
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img)
        return img, row['label']

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size*4

    images, labels = zip(*sources)
    
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images), 'label': list(labels)}) 

    if training:
        ds = ds.shuffle(shuffle_buffer_size)
    
    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(lambda x,y: (preprocess_image(x, pixels), y))
    
    if training:
        ds = ds.map(lambda x,y: (augment_image(x), y))
        
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds

def show3(batch):
    label_batch = batch[1].numpy()
    image_batch = batch[0].numpy()
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        axarr[i].set(xlabel='label = {}'.format(label_batch[i]))

def augmentImage(image):
    return image

def drawResult(H, N, val = False):
    fig, axs = plt.subplots(2)
    fig.suptitle('Training Loss and Accuracy on Dataset')
    axs[0].plot(np.arange(0, N), H.history["loss"], label="train_loss")
    if(val):
        axs[0].plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    axs[0].set_xlabel("Epoch #")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="lower left")
    axs[1].plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    if(val):
        axs[1].plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    axs[1].set_xlabel("Epoch #")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc="lower left")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset")

    parser.add_argument('--dir', '-d', help="directory with images")
    
    args = parser.parse_args()
    prepare_dataset(args.dir)
