import os
import sys
import json 
import random
import numpy as np
import pandas as pd
from collections import defaultdict

def prepareDataset(dataDir):
    if not os.path.exists('image_files'):
        os.makedirs('image_files')
    
    annDir = os.path.join(dataDir, 'ann')
    annList = [file for file in os.listdir(annDir) if os.path.isfile(os.path.join(annDir, file))]
    metada = defaultdict(list)
    fn = lambda x: str(hash(x) % ((sys.maxsize + 1) * 2)) + '.PNG'
    for ann in annList:
        with open(os.path.join(annDir, ann),'r') as f:
            dataStore = json.loads(f.read())
        for i in dataStore['objects']:
            newFileName = fn(ann)
            metada['imageName'].append(newFileName)
            metada['imageLabel'].append(i['tags'][0]['name'])

def saveCropImage(filePath, newName, exterior):
     



prepareDataset(r'C:\Users\USER\Desktop\Universidad\Grupo de estudio DeepLearning\ART_DATASET\Dataset')