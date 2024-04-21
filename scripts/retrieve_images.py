import os
import json

import numpy as np
import argparse
import pickle

import cv2

def get_images_from_mp4s():
    # directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    trajectory_dir = os.path.join(cur_dir, "../trajectories")

    # images npy array
    images = []

    # go through 0-0-0-0, 0-0-0-2, ..., 2-2-2-2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    variant = str(i) + "-" + str(j) + "-" + str(k) + "-" + str(l)
                    for m in range (4):
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
                            if success:
                                images.append(image)
                            count += 1
                        print("Done variant: ", variant, "trial: ", m, "frames: ", count)


    # save images as npy
    image_dir = os.path.join(cur_dir, "dataset")
    np.save(os.path.join(image_dir, "images.npy"), images

if __name__ == '__main__':
    print("-->>-->>-- Retrieving images from mp4s! --<<--<<--")

    get_images_from_mp4s()