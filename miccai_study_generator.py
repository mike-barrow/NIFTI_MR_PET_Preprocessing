# data preparation (conversion of DICOM PET/CT studies to nifti format for running automated lesion segmentation)

# run script from command line as follows:
# python tcia_dicom_to_nifti.py /PATH/TO/DICOM/FDG-PET-CT-Lesions/ /PATH/TO/NIFTI/FDG-PET-CT-Lesions/

# you can ignore the nilearn warning:
# .../nilearn/image/resampling.py:527: UserWarning: Casting data from int16 to float32 warnings.warn("Casting data from %s to %s" % (data.dtype.name, aux)) 
# or run as python -W ignore tcia_dicom_to_nifti.py /PATH/TO/DICOM/FDG-PET-CT-Lesions/ /PATH/TO/NIFTI/FDG-PET-CT-Lesions/

import pathlib as plb
#import tempfile
import os
#import dicom2nifti
import nibabel as nib
import numpy as np
#import pydicom
import sys
import shutil
import nilearn.image
from tqdm import tqdm
import re
#import scipy
import itertools
import pprint
#from sklearn.preprocessing import normalize

class SeqStudy:
    def SeqStudy(self, n, modalities, allcases):
        self.dirs = []
        self.case_count = n
        self.modalities = modalities

        for c in allcases:
            mc = identify_modalities(c)
            for m in modalities:
                if m not in mc:
                    break
                else:
                    self.cases.append(c)
                    n = n -1
            if n == 0:
                break
        pass

def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)

def find_studies(path_to_data):
    # find all studies
    nii_root = plb.Path(path_to_data)
    patient_dirs = list(nii_root.glob('*'))
    return patient_dirs
    
#find all known nifti modalities. These should have specific names to be identified.
def identify_modalities(d):
    fs = list(plb.Path(d).glob('*nii.gz'))
    ms = set()
    for f in fs:
        ms.add(f.stem.split('.')[0].replace("res",""))#Add only file name without any extension or "res" part to set.
    return list(ms)

def resample(srcf,tgtf,fill_value=-1024):
    s = nib.load(srcf)
    t = nib.load(tgtf)
    ss = nilearn.image.resample_to_img(source_img=s,target_img=t,fill_value=fill_value)
    outf = srcf.parent/str(srcf.stem.split('.')[0]+'res.nii.gz')
    nib.save(ss,outf)

def resample_mr(nii_out_path):
    # resample CT to PET resolution
    if not os.path.isfile(nii_out_path/'T1.nii.gz'):
        return      #no MR
    #check existance of all other types of modality
    t1f = nii_out_path/'T1.nii.gz'
    t2f = nii_out_path/'T2.nii.gz'
    dwi1f = nii_out_path/'DWI1.nii.gz'
    dwi2f= nii_out_path/'DWI2.nii.gz'
    ptf = nii_out_path/'PET.nii.gz'
    MR = [t1f,t2f,dwi1f,dwi2f] #does not work until the affine is sorted out
    for src in MR:
        if os.path.isfile(src):
            resample(src,ptf)

        


def resample_mask(nii_out_path):
    # resample CT to PET resolution
    if not os.path.isfile(nii_out_path/'SEG.nii.gz'):
        return      #no Mask (impossible?)
    seg   = nib.load(nii_out_path/'SEG.nii.gz')
    pet  = nib.load(nii_out_path/'PET.nii.gz')
    #always effs upCTres = nilearn.image.resample_to_img(seg, pet, fill_value=0)
    aff= pet.affine[0:3,0:3]    #these should be co-registered so eff the 4x4 affine. rescale only.
    poo = nilearn.image.resample_img(seg, target_affine=pet.affine, target_shape=pet.shape, fill_value=0,interpolation='nearest')
    if not poo.shape == pet.shape:
        CTres = nilearn.image.resample_to_img(poo, pet, fill_value=0,interpolation='nearest')
        nib.save(CTres, nii_out_path/'SEGres.nii.gz')
        print(f"{nii_out_path}weird ?")
    else:
        nib.save(poo, nii_out_path/'SEGres.nii.gz')


def convert_tcia_to_nifti(study_dirs,nii_out_root):
    # batch conversion of all patients
    for study_dir in tqdm(study_dirs):
        
        patient = study_dir.parent.name
        print("The following patient directory is being processed: ", patient)

        modalities = identify_modalities(study_dir)
        nii_out_path = plb.Path(nii_out_root/study_dir.parent.name)
        nii_out_path = nii_out_path/study_dir.name
        
        os.makedirs(nii_out_path, exist_ok=True)

        resample_mask(nii_out_path)
        resample_mr(nii_out_path)
        



if __name__ == "__main__":
    

    
    path_to_data = plb.Path(sys.argv[1])  # path to downloaded TCIA DICOM database, e.g. '.../FDG-PET-CT-Lesions/'
    DUT_root = plb.Path(sys.argv[2])  # path to the to be created NiFTI files, e.g. '...tcia_nifti/FDG-PET-CT-Lesions/')
    
    #TODO take in actual path
    path_to_data = '/media/king/4TB_B/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Known_Good_Lymphoma_Segs_Feb25'

    study_dirs = find_studies(path_to_data)

    #Get matrix of modalities
    available_modalities = {}
    modality_matrix = {}
    for d in study_dirs:
        dm =  identify_modalities(d)
        modality_matrix[d] = dm
        for k in dm:
            if k not in available_modalities:
                available_modalities[k] = 1
            else:
                available_modalities[k] = available_modalities[k] + 1
    del(available_modalities['SEG'])
    del(available_modalities['PET'])    #sufficient to use only thresholded PET(SUV)
    print(f'Available Sequences: {available_modalities}')
    #define all combinations.
    #start 1's, then 2's... stop when you have performance desired and drop bottom k each time?
    C = []
    cn = 3  #max elements in a combination  
    print(f'Possible combinations: 1->{cn} elements')
    for i in range(1,cn+1):
        c = list(itertools.combinations(available_modalities.keys(),i))
        print(f'{i} elements:')
        print(c)
        C.append(c) 

    #create viable set sizes for each permutation
    c_cohort = []

    minlimit = 25
    for c in C[0]:
        for M in c: #for all modalities in one given permutation
            cl = [available_modalities[m] for m in M]
    limit = min(cl)                               #max size of this study set
    print(f"dataset size is {limit} as compared with threshold: {minlimit}")
    seq_study = SeqStudy(limit,M,study_dirs)

    #create sub-sample sets

    #define nnUNet tasks

    #train?

    #convert_tcia_to_nifti(candidate_dirs, nii_out_root)
    #convert_tcia_to_nifti(study_dirs, nii_out_root)
    exit()
