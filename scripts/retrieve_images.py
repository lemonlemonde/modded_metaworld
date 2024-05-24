import os
import json

import numpy as np
import argparse
import pickle

import cv2
import torch
from torchvision.transforms import v2
from PIL import Image
from einops import rearrange

def get_images_from_mp4s():
    # directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    trajectory_dir = os.path.join(cur_dir, "../trajectories")

    batches = []

    # error
    defects = []

    flag = False

    # go through 0-0-0-0, 0-0-0-2, ..., 2-2-2-2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    variant = str(i) + "-" + str(j) + "-" + str(k) + "-" + str(l)
                    for m in range (4):
                        images = []
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

                            center = image.shape
                            w = 400
                            h = 400
                            x = center[1]/2 - w/2
                            y = center[0]/2 - h/2

                            crop_img = image[int(y):int(y+h), int(x):int(x+w)]
                            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                            # resize 
                            half = cv2.resize(crop_img, (224, 224))

                            images.append(half)

                            if (not flag):
                                # save image
                                image_dir = os.path.join(cur_dir, "../dataset", "images")
                                if not os.path.exists(image_dir):
                                    os.makedirs(image_dir)
                                cv2.imwrite(os.path.join(image_dir, variant + "_" + str(m) + "_" + str(count) + ".jpg"), half)
                                flag = True

                            count += 1
                        print("Done variant:", variant, " trial:", m, " frames:", count)
                        if (count != 501):
                            print("Variant:", variant, " trial:", m, " doesn't have 501 frames!!!!!")
                            defects.append("variant:" + str(variant) + " trial: " + str(m))

                        images.pop()
                        batches.append(np.stack(images))

    for i in defects:
        print(defects)

    # save images as npy
    image_dir = os.path.join(cur_dir, "../dataset")
    # print("before:")
    # print(images[0])
    # images = np.insert(images, 0, 1, axis=-1)
    # print("after:")
    # print(images[0])
    # images = np.array(images)
    # images = images[np.newaxis, :]
    batches = np.stack(batches)
    print("batch shape")
    print(batches.shape)
    np.save(os.path.join(image_dir, "images.npy"), batches)

if __name__ == '__main__':
    print("-->>-->>-- Retrieving images from mp4s! --<<--<<--")

    get_images_from_mp4s()