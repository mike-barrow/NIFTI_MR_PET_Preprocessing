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
 "numTraining": 1, 
 "numTest": 1,
"""
"""
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

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

#get all cases from preprocessed folder
taskname = "T1PETOnly"
tasknumber = "600"
infolder = "/home/king/Data/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Task600_MICCAI_Data"
outbase = "/home/king/Data/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Task"+tasknumber+"_" + taskname 
outtr = outbase + "/imagesTr"
outlb = outbase + "/labelsTr"
outval = outbase + "/imagesTs"
dsetf = outbase + "/dataset.json"
cases = []

for root,dirs,files in os.walk(infolder):
    if not dirs:
        cases.append(root)

CTres = 0
PET = 1

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

f = outtr   #folder to store training set of images
#see format expected for dataset here: https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md 
tr_ims = []
val_ims = []
for i, d in enumerate(cases):
    print(d)
    #seg = 0
    #CTres = 3
    #SUV = 4
    #print(os.listdir(c))

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
        #f = outval
        #print(f'Debug_{i:03d}_{CTres:04d}.nii.gz')
        fs = d + "/" + os.listdir(d)[2]#MICCAI24, T1#TCIA[3] 
        fd = outtr + f'/{taskname}_{i:03d}_{0:04d}.nii.gz'
        shutil.copy(fs,fd)
        fs = d + "/" + os.listdir(d)[4]#MICCAI24, SUV#TCIA[4] 
        fd = outtr + f'/{taskname}_{i:03d}_{1:04d}.nii.gz'
        shutil.copy(fs,fd)
        fs = d + "/" + os.listdir(d)[3]#MICCAI24 seg(res)#TCIA[0] 
        fd = outlb + f'/{taskname}_{i:03d}.nii.gz'
        shutil.copy(fs,fd)
    else:                           #copy files to validation folder 
        fs = d + "/" + os.listdir(d)[2]#MICCAI24#TCIA[3] 
        fd = outval + f'/{taskname}_{i:03d}_{0:04d}.nii.gz'
        shutil.copy(fs,fd)
        fs = d + "/" + os.listdir(d)[4]#MICCAI24#TCIA[4] 
        fd = outval + f'/{taskname}_{i:03d}_{1:04d}.nii.gz'
        shutil.copy(fs,fd)

    

#cat the header and training files into a dataset for nnU-Net
nnU_Net_data["training"] = tr_ims
nnU_Net_data["test"] = val_ims 
d = nnU_Net_dataset + ''.join(json.dumps(nnU_Net_data, indent=4).splitlines(keepends=True)[1:]) + '\n'

#write the data to file
with open(dsetf,'w') as f:
    f.write(d)