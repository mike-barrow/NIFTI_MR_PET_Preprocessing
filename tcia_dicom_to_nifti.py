# data preparation (conversion of DICOM PET/CT studies to nifti format for running automated lesion segmentation)

# run script from command line as follows:
# python tcia_dicom_to_nifti.py /PATH/TO/DICOM/FDG-PET-CT-Lesions/ /PATH/TO/NIFTI/FDG-PET-CT-Lesions/

# you can ignore the nilearn warning:
# .../nilearn/image/resampling.py:527: UserWarning: Casting data from int16 to float32 warnings.warn("Casting data from %s to %s" % (data.dtype.name, aux)) 
# or run as python -W ignore tcia_dicom_to_nifti.py /PATH/TO/DICOM/FDG-PET-CT-Lesions/ /PATH/TO/NIFTI/FDG-PET-CT-Lesions/

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
import re
import scipy
from sklearn.preprocessing import normalize

def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)

def find_studies(path_to_data):
    # find all studies
    dicom_root = plb.Path(path_to_data)
    patient_dirs = list(dicom_root.glob('*'))
    #DEBUG
    return patient_dirs
    #END DEBUG
    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        #print(sub_dirs)
        study_dirs.extend(sub_dirs)
        
        #dicom_dirs = dicom_dirs.append(dir.glob('*'))
    return study_dirs


def identify_modalities(study_dir):
    # identify CT, PET and mask subfolders and return dicitionary of modalities and corresponding paths, also return series ID, output is a dictionary
    study_dir = plb.Path(study_dir)
    sub_dirs = list(study_dir.glob('*'))

    modalities = {}

    for dir in sub_dirs:
        if next(dir.glob('*.dcm'),False):
            first_file = next(dir.glob('*.dcm'))
            ds = pydicom.dcmread(str(first_file))
            #print(ds)
            #modality = ds.Modality
            if 'MR' in str(dir) or 'T1' in str(dir):
                modality = 'T1'
            elif 'T2' in str(dir):
                modality = 'T2'
            elif 'PT' in str(dir):
                modality = 'PET'
            elif 'DWI1' in str(dir):
                modality = 'DWI1'
            elif 'DWI2' in str(dir):
                modality = 'DWI2'
            elif '_A' in str(dir):
                modality = 'SEG'
            modalities[modality] = dir
            modalities["ID"] = ds.StudyInstanceUID  #HACK see below hack why this is here.
        elif next(dir.glob('*.nii.gz'),False):                  #HACK we should not assume labels. have to rn.
            modalities['SEGnii'] = dir 
        elif next(glob_re(r"([^\.])",os.listdir(dir)),False):     #HACK our dataset has a bunch of files without dcm ext
            first_file = next(dir.glob('*'))
            ds = pydicom.dcmread(str(first_file))
            #print(ds)
            #modality = ds.Modality
            if 'MR' in str(dir) or 'T1' in str(dir):
                modality = 'T1'
            elif 'T2' in str(dir):
                modality = 'T2'
            elif 'PT' in str(dir):
                modality = 'PET'
            elif 'DWI1' in str(dir):
                modality = 'DWI1'
            elif 'DWI2' in str(dir):
                modality = 'DWI2'
            elif '_A' in str(dir):
                modality = 'SEG'
            modalities[modality] = dir
            modalities["ID"] = ds.StudyInstanceUID           
    
    #modalities["ID"] = ds.StudyInstanceUID
    return modalities

def dcm2nii_MR(MR_dcm_path, nii_out_path,sequence='T1'):
    # conversion of MR DICOM (in the MR_dcm_path) to nifti and save in nii_out_path
    with tempfile.TemporaryDirectory() as tmp: #convert MR       
        tmp = plb.Path(str(tmp))
        # convert dicom directory to nifti
        # (store results in temp directory)
        dicom2nifti.convert_directory(MR_dcm_path, str(tmp), 
                                    compression=True, reorient=True)
        nii = next(tmp.glob('*nii.gz'))
        # copy niftis to output folder with consistent naming
        shutil.copy(nii, nii_out_path/(sequence+'.nii.gz'))

def dcm2nii_CT(CT_dcm_path, nii_out_path):
    # conversion of CT DICOM (in the CT_dcm_path) to nifti and save in nii_out_path
    with tempfile.TemporaryDirectory() as tmp: #convert CT
        tmp = plb.Path(str(tmp))
        # convert dicom directory to nifti
        # (store results in temp directory)
        dicom2nifti.convert_directory(CT_dcm_path, str(tmp), 
                                      compression=True, reorient=True)
        nii = next(tmp.glob('*nii.gz'))
        # copy niftis to output folder with consistent naming
        shutil.copy(nii, nii_out_path/'CT.nii.gz')


def dcm2nii_PET(PET_dcm_path, nii_out_path):
    # conversion of PET DICOM (in the PET_dcm_path) to nifti (and SUV nifti) and save in nii_out_path
    first_pt_dcm = next(PET_dcm_path.glob('*'))#next(PET_dcm_path.glob('*.dcm'))
    suv_corr_factor = calculate_suv_factor(first_pt_dcm)
    with tempfile.TemporaryDirectory() as tmp: #convert PET
        tmp = plb.Path(str(tmp))
        # convert dicom directory to nifti
        # (store results in temp directory)
        dicom2nifti.convert_directory(PET_dcm_path, str(tmp), 
                                    compression=True, reorient=True)
        nii = next(tmp.glob('*nii.gz'))
        # copy nifti to output folder with consistent naming
        shutil.copy(nii, nii_out_path/'PET.nii.gz')
        # convert pet images to quantitative suv images and save nifti file
        suv_pet_nii = convert_pet(nib.load(nii_out_path/'PET.nii.gz'), suv_factor=suv_corr_factor)
        nib.save(suv_pet_nii, nii_out_path/'SUV.nii.gz')


def conv_time(time_str):
    # function for time conversion in DICOM tag
    return (float(time_str[:2]) * 3600 + float(time_str[2:4]) * 60 + float(time_str[4:13]))


def calculate_suv_factor(dcm_path):
    # reads a PET dicom file and calculates the SUV conversion factor
    ds = pydicom.dcmread(str(dcm_path))
    total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    acq_time = ds.AcquisitionTime
    weight = ds.PatientWeight
    time_diff = conv_time(acq_time) - conv_time(start_time)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)
    suv_factor = 1000 * weight / act_dose
    return suv_factor


def convert_pet(pet, suv_factor):
    # function for conversion of PET values to SUV (should work on Siemens PET/CT)
    affine = pet.affine
    pet_data = pet.get_fdata()
    pet_suv_data = (pet_data*suv_factor).astype(np.float32)
    pet_suv = nib.Nifti1Image(pet_suv_data, affine)
    return pet_suv


def dcm2nii_mask(mask_dcm_path, nii_out_path):
    # conversion of the mask dicom file to nifti (not directly possible with dicom2nifti)
    mask_dcm = list(mask_dcm_path.glob('*.dcm'))[0]
    mask = pydicom.read_file(str(mask_dcm))
    mask_array = mask.pixel_array
    
    """
    # get mask array to correct orientation (this procedure is dataset specific)
    mask_array = np.transpose(mask_array,(2,1,0) )  
    mask_orientation = mask[0x5200, 0x9229][0].PlaneOrientationSequence[0].ImageOrientationPatient
    if mask_orientation[4] == 1:
        mask_array = np.flip(mask_array, 1 )
    """
    

    # get affine matrix from the corresponding pet             
    pet = nib.load(str(nii_out_path/'PET.nii.gz'))
    pet_affine = pet.affine
    
    # return mask as nifti object
    mask_out = nib.Nifti1Image(mask_array, pet_affine)
    nib.save(mask_out, nii_out_path/'SEG.nii.gz')

#for existing masks in nii.gz, just move them. They have been co-registered a long time ago.
def process_existing_mask(mask_nii_in_path, nii_out_path):
    src_nii_gz = list(mask_nii_in_path.glob('*.nii.gz'))[0]
    shutil.copy2(str(src_nii_gz),str(nii_out_path/'SEG.nii.gz'))   
    
#TODO this down sampling is a bit brutal since PET is lower resolution than anything (mask and MR)
def resample_ct(nii_out_path):
    # resample CT to PET resolution
    if not os.path.isfile(nii_out_path/'CT.nii.gz'):
        return      #no CT
    ct   = nib.load(nii_out_path/'CT.nii.gz')
    pet  = nib.load(nii_out_path/'PET.nii.gz')
    CTres = nilearn.image.resample_to_img(ct, pet, fill_value=-1024)
    nib.save(CTres, nii_out_path/'CTres.nii.gz')

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
    #mr   = nib.load(nii_out_path/'T1.nii.gz')
    #pet  = nib.load(nii_out_path/'PET.nii.gz')
    #CTres = nilearn.image.resample_to_img(mr, pet, fill_value=-1024)
    #nib.save(CTres, nii_out_path/'T1res.nii.gz')
    #check existance of all other types of modality
    t1f = nii_out_path/'T1.nii.gz'
    t2f = nii_out_path/'T2.nii.gz'
    dwi1f = nii_out_path/'DWI1.nii.gz'
    dwi2f= nii_out_path/'DWI2.nii.gz'
    ptf = nii_out_path/'PET.nii.gz'
    MR = [t1f]#,t2f,dwi1f,dwi2f] #does not work until the affine is sorted out
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
    #poo = nilearn.image.resample_img(seg, target_affine=aff, fill_value=0,interpolation='nearest')
    poo = nilearn.image.resample_img(seg, target_affine=pet.affine, target_shape=pet.shape, fill_value=0,interpolation='nearest')
    if not poo.shape == pet.shape:
        CTres = nilearn.image.resample_to_img(poo, pet, fill_value=0,interpolation='nearest')
        nib.save(CTres, nii_out_path/'SEGres.nii.gz')
        print(f"{nii_out_path}weird ?")
    else:
        nib.save(poo, nii_out_path/'SEGres.nii.gz')

def tcia_to_nifti(tcia_path, nii_out_path, modality='CT'):
    # conversion for a single file
    # creates a nifti file for one patient/study
    # tcia_path:        path to a DICOM directory for a specific study of one patient
    # nii_out_path:     path to a directory where nifti file for one patient, study and modality will be stored
    # modality:         modality to be converted CT, PET or mask ('CT', 'PT', 'SEG')
    os.makedirs(nii_out_path, exist_ok=True)
    if modality == 'CT':
        dcm2nii_CT(tcia_path, nii_out_path)
        resample_ct(nii_out_path)
    elif modality == 'PET':
        dcm2nii_PET(tcia_path, nii_out_path)
    elif modality == 'SEG':
        dcm2nii_mask(tcia_path, nii_out_path)
        resample_mask(nii_out_path)
    elif modality == 'SEGnii':
        process_existing_mask(tcia_path, nii_out_path)
        resample_mask(nii_out_path)

#TODO this is not patched and also unclear why it exists.
def tcia_to_nifti_study(study_path, nii_out_path):
    # conversion for a single study
    # creates NIfTI files for one patient
    # study_path:       path to a study directory containing all DICOM files for a specific study of one patient
    # nii_out_path:     path to a directory where all nifti files for one patient and study will be stored
    study_path = plb.Path(study_path)
    modalities = identify_modalities(study_path)
    nii_out_path = plb.Path(nii_out_root / study_path.parent.name)
    nii_out_path = nii_out_path/study_path.name
    os.makedirs(nii_out_path, exist_ok=True)

    ct_dir = modalities["CT"]
    dcm2nii_CT(ct_dir, nii_out_path)

    mr_dir = modalities["MR"]
    dcm2nii_MR(mr_dir, nii_out_path)

    pet_dir = modalities["PT"]
    dcm2nii_PET(pet_dir, nii_out_path)

    #seg_dir = modalities["SEG"]
    #dcm2nii_mask(seg_dir, nii_out_path)

    #seg_nii_dir = modalities["SEGnii"]
    #process_existing_mask(seg_nii_dir, nii_out_path)

    resample_ct(nii_out_path)
    resample_mr(nii_out_path)
    #resample_mask(nii_out_path)


def convert_tcia_to_nifti(study_dirs,nii_out_root):
    # batch conversion of all patients
    for study_dir in tqdm(study_dirs):
        
        patient = study_dir.parent.name
        print("The following patient directory is being processed: ", patient)

        modalities = identify_modalities(study_dir)
        nii_out_path = plb.Path(nii_out_root/study_dir.parent.name)
        nii_out_path = nii_out_path/study_dir.name
    
        os.makedirs(nii_out_path, exist_ok=True)
        
        if 'CT' in modalities:
            ct_dir = modalities["CT"]
            dcm2nii_CT(ct_dir, nii_out_path)

        #TODO more refined for DWI T1, T2, etc.
        if 'T1' in modalities:
            mr_dir = modalities["T1"]
            dcm2nii_MR(mr_dir, nii_out_path,'T1')
        """    
        if 'T2' in modalities:
            mr_dir = modalities["T2"]
            dcm2nii_MR(mr_dir, nii_out_path,'T2')
        if 'DWI1' in modalities:
            mr_dir = modalities["DWI1"]
            dcm2nii_MR(mr_dir, nii_out_path,'DWI1')            
        if 'DWI2' in modalities:
            mr_dir = modalities["DWI2"]
            dcm2nii_MR(mr_dir, nii_out_path,'DWI2')
        """
        if 'PET' in modalities:
            pet_dir = modalities["PET"]
            dcm2nii_PET(pet_dir, nii_out_path)
        
        
        if 'SEG' in modalities:
            seg_dir = modalities["SEG"]
            dcm2nii_mask(seg_dir, nii_out_path)
            #resample_mask(nii_out_path)

        #TODO deal with existing nii masks.
        if 'SEGnii' in modalities:
            seg_nii_dir = modalities["SEGnii"]
            process_existing_mask(seg_nii_dir, nii_out_path)
            #resample_mask(nii_out_path)
         
        #Doesn't need to be done for every case, and does not handle every modality actually
        fixxfrm(nii_out_path,'T1.nii.gz')

        if 'SEG' in modalities or 'SEGnii' in modalities:
            resample_mask(nii_out_path)

        resample_ct(nii_out_path)
        resample_mr(nii_out_path)
        


def closest_unitary(A): 
    """ Calculate the unitary matrix U that is closest with respect to the 
        operator norm distance to the general matrix A. 
 
        Return U as a numpy matrix. 
    """ 
    V, __, Wh = scipy.linalg.svd(A) 
    U = np.matrix(V.dot(Wh)) 
    return U 

def compute_rot(A): 
    """ Calculate the unitary matrix U that is closest with respect to the 
        operator norm distance to the general matrix A. 
 
        Return U as a numpy matrix. 
    """ 
    V, __, Wh = scipy.linalg.svd(A) 
    R = V*Wh 
    return R 
from scipy.spatial.transform import Rotation as R


def affine2d(ds):
    F11, F21, F31 = ds.ImageOrientationPatient[3:]
    F12, F22, F32 = ds.ImageOrientationPatient[:3]

    dr, dc = ds.PixelSpacing
    Sx, Sy, Sz = ds.ImagePositionPatient

    return np.array(
        [
            [F11 * dr, F12 * dc, 0, Sx],
            [F21 * dr, F22 * dc, 0, Sy],
            [F31 * dr, F32 * dc, 0, Sz],
            [0, 0, 0, 1]
        ]
    )

#debug
#for some reason, some nifti files get created with "non orthonormal" transfor matrices for the MR.
#who cares? well, the nifti readers apparently. The fix seems to be to remove translation from MR and subtract from PET (give common offset).
#Tried so many other things with no luck.
def fixxfrm(nii_path,nii_file):
    #idea 6702-500 just subtract affine part of transform into PET. Zero out affine in MR
    mr = nib.load(str(nii_path/'T1.nii.gz'))
    mraffine = mr.affine
    
    #PET and SUV must be corrected.
    pet = nib.load(str(nii_path/'PET.nii.gz')) 
    petaffine = pet.affine
    petaffine[:3,3] = petaffine[:3,3] - mraffine[:3,3] #subtract MR translation from PET translation  
    petd = np.asanyarray(pet.dataobj)
    nipet = nib.Nifti1Image(petd,petaffine)   
    nib.save(nipet, nii_path/('PET.nii.gz'))
    #Now SUV
    pet = nib.load(str(nii_path/'SUV.nii.gz')) 
    petaffine = pet.affine
    petaffine[:3,3] = petaffine[:3,3] - mraffine[:3,3] #subtract MR translation from SUV translation  
    petd = np.asanyarray(pet.dataobj)
    nipet = nib.Nifti1Image(petd,petaffine)   
    nib.save(nipet, nii_path/('SUV.nii.gz'))

    mraffine[:3,3] = [0,0,0]    #nuke MR translation, which fixes non-orthonormal junk when trying to open.
    mrd = np.asanyarray(mr.dataobj)
    nimr = nib.Nifti1Image(mrd,mraffine)
    nib.save(nimr, nii_path/('T1.nii.gz'))
    """
    does not work properly
    mrmods = ['T1.nii.gz','T2.nii.gz','DWI1.nii.gz','DWI2.nii.gz']
    for mf in mrmods:
        mfn = nii_path/mf
        if os.path.isfile(mfn):
            mr = nib.load(mfn)
            mrd = np.asanyarray(mr.dataobj)
            nimr = nib.Nifti1Image(mrd,mraffine)
            nib.save(nimr, mfn)
    """
    
    #fix annotations for this file (if exists)
    if os.path.isfile(str(nii_path/'SEG.nii.gz')):
        seg = nib.load(str(nii_path/'SEG.nii.gz')) 
        segd = np.asanyarray(seg.dataobj)
        niseg = nib.Nifti1Image(segd,mraffine)   
        nib.save(niseg, nii_path/('SEG.nii.gz'))

    return
    #idea 6702-438 just construct affine from dicom header yourself.
    #dicomp = plb.Path('/media/king/4TB_B/Lymphoma_Data_Cache/MICCAI24/Unlabeled_cases_Feb_5_2024/ST_S3_33_BL/ST_S3_33_BL_MR/')
    #ds = pydicom.dcmread(str(dicomp))
    mr = nib.load(str(nii_path/'T1.nii.gz'))
    #redux rotations....
    rot = R.from_matrix(mr.affine[:3,:3])
    #mraffine = mr.affine
    mraffine = np.zeros([4,4])
    mraffine[:3,:3] = rot.as_matrix() 
    mraffine[3,3] = 1 
    #bummer with orthonormal

    mraffine = normalize(mraffine, axis=0, norm='l1')#most important step seems to do with l1norm of  translation part...
    #fix scale
    mraffine[0,0]*=0.7422#x (y deta) 0.7422 vs 2.60417
    mraffine[1,1]*=0.7422#y (x delta)0.7422 vs 2.60417
    mraffine[2,2]*=1.30143 #z (spacing delta)1.30143 vs 2.78 
    #fix translation
    #re-introduce translation (ratios seem messed up some how)
    mraffine[:,3] = mr.affine[:,3]
    mraffine[0,3]/=mr.affine[0,0]
    mraffine[1,3]/=mr.affine[1,1]
    mraffine[2,3]/=mr.affine[2,2]
    mraffine[0,3]*=0.7422
    mraffine[1,3]*=0.7422
    mraffine[2,3]*=1.30143  
    #mraffine[3,3] = (-1*sum(mraffine[:3,3])) +1

    mrd = np.asanyarray(mr.dataobj)
    nimr = nib.Nifti1Image(mrd,mraffine)
    nib.save(nimr, nii_path/('fixed_MR.nii.gz'))

    pet = nib.load(str(nii_path/'PET.nii.gz')) 
    rot = R.from_matrix(pet.affine[:3,:3])
    petaffine = np.zeros([4,4])#pet.affine
    petaffine[:3,:3] = rot.as_matrix()
    petaffine[3,3] = 1 

    petaffine = normalize(petaffine, axis=0, norm='l1')#most important step seems to do with l1norm of  translation part...
    #works without this but is too small...
    #have to scale the pet by the delta of voxel size
    petaffine[0,0]*=2.60417#x (y deta) 0.7422 vs 2.60417
    petaffine[1,1]*=2.60417#y (x delta)0.7422 vs 2.60417
    petaffine[2,2]*=2.78#z (spacing delta)1.30143 vs 2.78
    #re-introduce translation (ratios seem messed up some how)
    #bummer with orthonormal
    #fix translation
    #re-introduce translation (ratios seem messed up some how)
    petaffine[:,3] = pet.affine[:,3]
    petaffine[0,3]/=pet.affine[0,0]
    petaffine[1,3]/=pet.affine[1,1]
    petaffine[2,3]/=pet.affine[2,2]
    petaffine[0,3]*=2.60417
    petaffine[1,3]*=2.60417
    petaffine[2,3]*=2.78 
    petd = np.asanyarray(pet.dataobj)
    nipet = nib.Nifti1Image(petd,petaffine)   
    nib.save(nipet, nii_path/('fixed_PET.nii.gz'))

    return
    #idea 5702-438
    mr = nib.load(str(nii_path/'T1.nii.gz'))
    #redux rotations....
    rot = R.from_matrix(mr.affine[:3,:3])
    mraffine = mr.affine
    mraffine[:3,:3] = rot.as_matrix()
    mraffine = normalize(mraffine, axis=0, norm='l1')#most important step seems to do with l1norm of  translation part...
    mrd = np.asanyarray(mr.dataobj)
    nimr = nib.Nifti1Image(mrd,mraffine)
    nib.save(nimr, nii_path/('fixed_MR.nii.gz'))

    pet = nib.load(str(nii_path/'PET.nii.gz')) 
    rot = R.from_matrix(pet.affine[:3,:3])
    petaffine = pet.affine
    petaffine[:3,:3] = rot.as_matrix()
    petaffine = normalize(petaffine, axis=0, norm='l1')#most important step seems to do with l1norm of  translation part...
    #works without this but is too small...
    #have to scale the pet by the delta of voxel size
    petaffine[0,0]*=2.60417/0.7422#x (y deta) 0.7422 vs 2.60417
    petaffine[1,1]*=2.60417/0.7422#y (x delta)0.7422 vs 2.60417
    petaffine[2,2]*=2.78/1.30143 #z (spacing delta)1.30143 vs 2.78
    petd = np.asanyarray(pet.dataobj)
    nipet = nib.Nifti1Image(petd,petaffine)   
    nib.save(nipet, nii_path/('fixed_PET.nii.gz'))

    return
    """
    #Idea 4567
    mr = nib.load(str(nii_path/'T1.nii.gz'))
    pet = nib.load(str(nii_path/'PET.nii.gz'))
    U = mr.affine[:3,:3]
    V = normalize(mr.affine[:3,:3], axis=0, norm='l1') #will be the new affine for mr
    U1 = np.linalg.inv(U)                        #use to compyte T for pet
    T = V*U1
    #take l1 norm of mr
    #compute the xform to go from original MR affine to l1 norm
    #apply the xform to the PET affine
    #they should both be in the same space
    mraffine = mr.affine
    mraffine[:3,:3] = V[:3,:3]
    petaffine = pet.affine
    petaffine[:3,:3] =  T*pet.affine[:3,:3]
    petd = np.asanyarray(pet.dataobj)
    nipet = nib.Nifti1Image(petd,petaffine)
    nib.save(nipet, nii_path/('fixed_PET.nii.gz'))
    """
    mrd = np.asanyarray(mr.dataobj)
    nimr = nib.Nifti1Image(mrd,mraffine)
    nib.save(nimr, nii_path/('fixed_MR.nii.gz'))    
    return
    #wway to fix this seems to be to give both images the same diagonal in their affine matrix. Just works. Shrug.
    mr = nib.load(str(nii_path/'T1.nii.gz'))
    pet = nib.load(str(nii_path/'PET.nii.gz'))
    petaffine = np.zeros([4,4])
    np.fill_diagonal(petaffine,pet.affine.diagonal())
    #mraffine = np.zeros([4,4])
    #np.fill_diagonal(mraffine,mr.affine.diagonal())
    mraffine = mr.affine
    mraffine[1,0] = 0; mraffine[2,0]=0; mraffine[2,1]=0
    mraffine[0,1] = 0; mraffine[0,2]=0; mraffine[2,2]=0
    mrd = np.asanyarray(mr.dataobj)
    petd = np.asanyarray(pet.dataobj)
    nipet = nib.Nifti1Image(petd,petaffine)
    nimr = nib.Nifti1Image(mrd,mraffine)
    nib.save(nipet, nii_path/('fixed_PET.nii.gz'))
    nib.save(nimr, nii_path/('fixed_MR.nii.gz'))
    return

    mr = nib.load(str(nii_path/nii_file))
    d = np.asanyarray(mr.dataobj)
    #ni = nib.Nifti1Image(d,np.identity(4))#nuke the old affine
    ni = nib.Nifti1Image(d,closest_unitary(mr.affine))#try to correct old affine
    #Try to correct the affine transform.
    nib.save(ni, nii_path/('fixed_'+str(nii_file)))


#keepout list, current cases known to have bad image data in the first place
blacklist = ['ST_S1_8','ST_S1_14','ST_S1_50','ST_S2_104','ST_S2_110','ST_S3_29','ST_S3_37' ,'ST_S3_38', #resample fail
             'ST_S1_21','ST_S1_22','ST_S1_17','ST_S1_40', 'ST_S1_89', 'ST_S1_88','ST_S3_29','ST_S1_31','ST_S1_23','ST_S1_37','ST_S1_45','ST_S1_16',#] #label fail
             'ST_S1_12','ST_S1_44','ST_S3_49']#processed already
blacklist = []
if __name__ == "__main__":
    

    
    path_to_data = plb.Path(sys.argv[1])  # path to downloaded TCIA DICOM database, e.g. '.../FDG-PET-CT-Lesions/'
    nii_out_root = plb.Path(sys.argv[2])  # path to the to be created NiFTI files, e.g. '...tcia_nifti/FDG-PET-CT-Lesions/')
    
    #see: https://github.com/icometrix/dicom2nifti we have bad affine matrices from our inconsistent slice increments. This may help.
    dicom2nifti.settings.disable_validate_slice_increment() #Do not know if this will mess things up but a problem for Case S1_2 at least
    #dicom2nifti.settings.enable_resampling()
    #dicom2nifti.settings.set_resample_spline_interpolation_order(1)
    #dicom2nifti.settings.set_resample_padding(-1000)
    study_dirs = find_studies(path_to_data)
    candidate_dirs = []
    for s in study_dirs:
        if not any([b in str(s) for b in blacklist]):
            candidate_dirs.append(s)
    #convert_tcia_to_nifti(candidate_dirs, nii_out_root)
    convert_tcia_to_nifti(study_dirs, nii_out_root)
    exit()
    #candidate_dirs = candidate_dirs[31:]
    """
    candidate_dirs = [plb.Path('/media/king/4TB_B/Lymphoma_Data_Cache/MICCAI24/Lymphoma_Cases_Feb_5_2024/ST_S1_12_BL'),
                      plb.Path('/media/king/4TB_B/Lymphoma_Data_Cache/MICCAI24/Lymphoma_Cases_Feb_5_2024/ST_S1_44_BL'),
                      plb.Path('/media/king/4TB_B/Lymphoma_Data_Cache/MICCAI24/Lymphoma_Cases_Feb_5_2024/ST_S3_49_BL')
                      ]
    """
    """
    #fix bad nifti files
    candidate_dirs = [plb.Path('/media/king/4TB_B/Lymphoma_Data_Cache/MICCAI24/Unlabeled_cases_Feb_5_2024/ST_S3_9_BL'),
                      plb.Path('/media/king/4TB_B/Lymphoma_Data_Cache/MICCAI24/Unlabeled_cases_Feb_5_2024/ST_S3_33_BL'),
                      plb.Path('/media/king/4TB_B/Lymphoma_Data_Cache/MICCAI24/Unlabeled_cases_Feb_5_2024/ST_S3_36_BL'),
                      plb.Path('/media/king/4TB_B/Lymphoma_Data_Cache/MICCAI24/Unlabeled_cases_Feb_5_2024/ST_S3_82_BL')
                      ]; nii_out_root =   plb.Path("/home/king/Data/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Lyphoma_Dataset_unprocessed"  )
    """
    #fix bad segmentation transforms NOTE: This does not fix badsegmentations at all. They remain flipped.
    p = plb.Path('/media/king/4TB_B/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Task600_MICCAI_Data/Lymphoma_Cases_Feb_5_2024/')
    study_dirs = [f for f in p.iterdir() if f.is_dir()]
    fix_dirs = []
    for s in study_dirs:
        if not any([b in str(s) for b in blacklist]):
            fix_dirs.append(s)
    cpts = [('_MR','T1.nii.gz'),('_PT','PET.nii.gz')]
    for p in fix_dirs:
        fixxfrm(p,cpts[0][1])
    exit()
    #end fix bad segmentation transforms        
 

    exit()
    #fix known bad DICOM orientations
    bd1 = '/media/king/4TB_B/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Task600_MICCAI_Data/Lymphoma_Cases_Feb_5_2024/'
    bd2 = '/media/king/4TB_B/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Lyphoma_Dataset_unprocessed/Unlabeled_cases_Feb_5_2024/'
    fix_dirs = [
                    plb.Path(bd1+'ST_S1_12_BL'),
                    plb.Path(bd1+'ST_S1_44_BL'),
                    plb.Path(bd1+'ST_S3_49_BL'),
                    plb.Path(bd2+'ST_S3_9_BL'),
                    plb.Path(bd2+'ST_S3_33_BL'),
                    plb.Path(bd2+'ST_S3_36_BL'),
                    plb.Path(bd2+'ST_S3_82_BL'),
                    ]   
    cpts = [('_MR','T1.nii.gz'),('_PT','PET.nii.gz')]
    for p in fix_dirs:
        #for f in cpts:
        #    fixxfrm(p,f[1])
        fixxfrm(p,cpts[0][1])
    exit()
