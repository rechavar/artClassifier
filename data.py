import os
import sys
import cv2
import json 
import random
import numpy as np
import pandas as pd

def prepareDataset(dataDir):
    if not os.path.exists('image_files'):
        os.makedirs('image_files')
    
    annDir = os.path.join(dataDir, 'ann')
    annList = [file for file in os.listdir(pascal_dir) if os.path.isfile(os.path.join(pascal_dir, file))]
    metada = defaultdict(list)
    fn = lambda x: str(hash(x) % ((sys.maxsize + 1) * 2)) + '.PNG'
    for ann in annList:
        jsonFile = json.dumps(ann)
        datastore = json.loads(jsonFile)
        
