#create nnunet dataset json. based on tci_dicom_to_nifti
#use the following format
nnU_Net_dataset = """
{ 
 "name": "Debug", 
 "description": "multi-modal debug only",
 "reference": "",
 "licence":"",
 "relase":"",
 "tensorImageSize": "3D",
 "modality": { 
   "0": "CTres",
   "1": "SUV"
 }, 
 "labels": { 
   "0": "background", 
   "1": "Lesions" 
 }, 
"""
"""
 "numTraining": 1, 
 "numTest": 1,
 "training":[{"image":"./imagesTr/Debug_000.nii.gz",
              "label":"./labelsTr/Debug_000.nii.gz"}], 
 "test": ["./imagesTs/Debug_001.nii.gz"]
 }
"""

from math import *
import pathlib as plb
import tempfile
import os
import dicom2nifti
import nibabel as nib
import numpy as np
import pydicom
import sys
import shutil
import nilearn.image
from tqdm import tqdm
import json

from tcia_dicom_to_nifti import *   #use a bunch of stuff here.
import re

#see:https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

#get all cases from preprocessed folder
taskname = "T1PETFeb20"
tasknumber = "611"
infolder = "/home/king/Data/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Known_Good_Lymphoma_Segs_Feb20"
outbase = "/home/king/Data/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Task"+tasknumber+"_" + taskname 
outtr = outbase + "/imagesTr"
outlb = outbase + "/labelsTr"
outval = outbase + "/imagesTs"
dsetf = outbase + "/dataset.json"
cases = []

for root,dirs,files in os.walk(infolder):
    if not dirs:
        cases.append(root)
#natural sort cases so that it is easy to map directory_of_cases <-> nnUNet indexing
cases = (sorted(cases, key=natural_sort_key))
print(f'found the following cases:')

nval = int(floor(len(cases)*0.1)) #TODO DEBUG ONELY
ntrn = int(len(cases) - nval)

#try to create directories for the dataset
maybe_mkdir_p(outbase)
maybe_mkdir_p(outtr)
maybe_mkdir_p(outlb)
maybe_mkdir_p(outval)

nnU_Net_data = { "training":[{"image":"./imagesTr/Debug_000.nii.gz",
              "label":"./labelsTr/Debug_000.nii.gz"}], 
 "test": ["./imagesTs/Debug_001.nii.gz"]}

ext = ".nii.gz"
#add more to this to create a training set with multi-modalities
inputs = ["T1res"+ext,"PET"+ext]#"SUV"+ext]
labels = ["SEGres"+ext]

f = outtr   #folder to store training set of images
#see format expected for dataset here: https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md 
tr_ims = []
val_ims = []
for i, d in enumerate(cases):

    #print(f'{taskname}_{i:03d}.nii.gz')
    c = f'./imagesTs/{taskname}_{i:03d}.nii.gz'

    #add case to dataset file
    if i < ntrn:
        tr_ims.append({"image":c,"label":c})
        #tr_lbs.append(c) 
    else:
        val_ims.append(c)

    #copy file
    if i < ntrn:                    #copy files to training folder
        print(f'TR: {taskname}_{i:03d} = {d}')
        for j,m in enumerate(inputs):
            fs = d + '/' + m
            fd = outtr + f'/{taskname}_{i:03d}_{j:04d}.nii.gz'
            shutil.copy(fs,fd)
        fs = d + "/" + labels[0]
        fd = outlb + f'/{taskname}_{i:03d}.nii.gz'
        shutil.copy(fs,fd)
    else:                           #copy files to validation folder 
        print(f'VAL : {taskname}_{i:03d} =  {d}')
        fs = d + "/" + labels[0]
        fd = outlb + f'/{taskname}_{i:03d}.nii.gz'
        shutil.copy(fs,fd)

    

#cat the header and training files into a dataset for nnU-Net
nnU_Net_data["training"] = tr_ims
nnU_Net_data["test"] = val_ims 
d = nnU_Net_dataset + str(f"\"numTraining\": {ntrn},\n\"numTest\": {nval},\n") + ''.join(json.dumps(nnU_Net_data, indent=4).splitlines(keepends=True)[1:]) + '\n'

#write the data to file
with open(dsetf,'w') as f:
    f.write(d)