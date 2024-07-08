# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:52:51 2024

@author: Laith Qushair
"""

import pandas as pd
import os
import shutil


# Dump all images into a folder and specify the path:
data_dir = os.getcwd() + "/data/All Images/"

# Path to destination directory where we want subfolders
dest_dir = os.getcwd() + "/data/Organized/"

# Read the csv file containing image names and corresponding labels
skin_df = pd.read_csv(os.getcwd() + '/data/HAM10000_metadata.csv')

label=skin_df['dx'].unique().tolist()  #Extract labels into a list
label_images = []


# Copy images to new folders
for i in label:
    os.mkdir(dest_dir + str(i) + "/")
    sample = skin_df[skin_df['dx'] == i]['image_id']
    label_images.extend(sample)
    for id in label_images:
        shutil.copyfile((data_dir + "/"+ id +".jpg"), (dest_dir + i + "/"+id+".jpg"))
    label_images=[]    
