import os
import json

import numpy as np
import argparse
import pickle

import cv2
import torch
from torchvision.transforms import v2
from PIL import Image

def get_images_from_mp4s():
    # directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    trajectory_dir = os.path.join(cur_dir, "../trajectories")

    # images npy array
    images = []

    # error
    defects = []

    # go through 0-0-0-0, 0-0-0-2, ..., 2-2-2-2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    variant = str(i) + "-" + str(j) + "-" + str(k) + "-" + str(l)
                    for m in range (4):
                        print("Retrieving images for variant: " + variant + " trial: " + str(m))
                        # trial
                        # get the mp4 from the directory
                        mp4_path = os.path.join(trajectory_dir, variant, "images_" + str(m), "button_press_" + str(m) + ".mp4")
                        vidObj = cv2.VideoCapture(mp4_path) 
                        
                        # counter var
                        count = 0
                        success = 1
                    
                        while success: 
                            # get frames
                            success, image = vidObj.read() 
                            if not success:
                                break

                            # resize 
                            # half = cv2.resize(crop_img, (0, 0), fx = 0.1, fy = 0.1)

                            center = image.shape
                            w = 224
                            h = 224
                            x = center[1]/2 - w/2
                            y = center[0]/2 - h/2

                            crop_img = image[int(y):int(y+h), int(x):int(x+w)]
                            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                            images.append(crop_img)

                            count += 1
                        print("Done variant: ", variant, "trial: ", m, "frames: ", count)
                        if (count != 502):
                            print("Variant: ", variant, " trial: ", m, " doesn't have 512 frames!!!!!")
                            defects.append("variant: ", variant, "trial: ", m)

    for i in defects:
        print(defects)

    # save images as npy
    image_dir = os.path.join(cur_dir, "../dataset")
    np.save(os.path.join(image_dir, "images.npy"), images)

if __name__ == '__main__':
    print("-->>-->>-- Retrieving images from mp4s! --<<--<<--")

    get_images_from_mp4s()