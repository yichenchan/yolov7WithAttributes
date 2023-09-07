# -*- coding: utf-8 -*-
import os
import sys
from imagededup.methods import PHash, CNN
from imagededup.utils import plot_duplicates
import cv2
import shutil
import numpy  as np

phasher1 = PHash()
phasher2 = CNN()


for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
    if not any(os.path.isdir(os.path.join(dirpath, d)) for d in os.listdir(dirpath)):

        print(dirpath)

        for d, n, fs in os.walk(dirpath):
            for f in fs:
                file = os.path.join(d, f)
                image = cv2.imread(file)
                if(image.shape[0] < 200 or image.shape[1] < 200):
                    os.remove(file)
                # #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # #mean = np.mean(image)
                # # if mean < 70:
                # if image.shape[1] != 1920 or image.shape[0] != 1080:
                #     #print(mean)
                #     shutil.copy(file, '/workspace/data/not_1920')
                #     #os.remove(file)


        # encodings = phasher2.encode_images(image_dir=dirpath)
        # duplicates = phasher2.find_duplicates_to_remove(encoding_map=encodings)

        # for dup in duplicates:
        #     print("removing " + dirpath + "/" + dup)
        #     os.remove(dirpath + '/' + dup)

