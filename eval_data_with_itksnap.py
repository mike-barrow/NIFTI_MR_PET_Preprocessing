#!/usr/bin/env python
# -*- coding: utf-8 -*-
#use this script to parse folder of data manually and check all .nii.gz files are properly formed
#pass in a directory to start the evaluation from. Assume: T1.nii.gz / T1res.nii.gz are main images SEG.nii.gz / SEGres.nii.gz are main segmentations
#you can specify additional modalities expected. If they are not present, mark them missing
#if they are bugged, mark them bugged
import tkinter as tk
import pathlib as plb
import sys
import re
import subprocess
import json
import nibabel as nib #just to get dimensions.

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(str(s))]

def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)

#get all nifti folders
def find_nifti_folders(path_to_data):
    nifti_root = plb.Path(path_to_data)
    nifti_dirs = list(nifti_root.glob('*[!json]'))
    return nifti_dirs


def save_report(pth,report):

    with open(pth, "w") as outfile: 
        json.dump(report, outfile)
            
    pass
#######################################################################
"""<?xml version="1.0" encoding="UTF-8" ?>
<!--ITK-SNAP (itksnap.org) Project File

This file can be moved/copied along with the images that it references
as long as the relative location of the images to the project file is 
the same. Do not modify the SaveLocation entry, or this will not work.
-->
<!DOCTYPE registry [
<!ELEMENT registry (entry*,folder*)>
<!ELEMENT folder (entry*,folder*)>
<!ELEMENT entry EMPTY>
<!ATTLIST folder key CDATA #REQUIRED>
<!ATTLIST entry key CDATA #REQUIRED>
<!ATTLIST entry value CDATA #REQUIRED>
]>
<registry>
  <entry key="SaveLocation" value="/home/king/Data/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Known_Good_Lymphoma_Segs_Feb27" />
  <entry key="Version" value="20230925" />
  <folder key="Annotations" >
    <entry key="Format" value="ITK-SNAP Annotation File" />
    <entry key="FormatDate" value="20150624" />
  </folder>
  <folder key="Layers" >
    <folder key="Layer[000]" >
      <entry key="AbsolutePath" value="T1.nii.gz" />
      <entry key="Role" value="MainRole" />
      <entry key="Tags" value="" />
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="255" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="0" />
        <entry key="Tags" value="" />
        <folder key="DisplayMapping" >
          <folder key="ColorMap" >
            <entry key="Preset" value="Grayscale" />
          </folder>
          <folder key="Curve" >
            <entry key="NumberOfControlPoints" value="3" />
            <folder key="ControlPoint[0]" >
              <entry key="tValue" value="0" />
              <entry key="xValue" value="0" />
            </folder>
            <folder key="ControlPoint[1]" >
              <entry key="tValue" value="0.5" />
              <entry key="xValue" value="0.5" />
            </folder>
            <folder key="ControlPoint[2]" >
              <entry key="tValue" value="1" />
              <entry key="xValue" value="1" />
            </folder>
          </folder>
        </folder>
      </folder>
      <folder key="ProjectMetaData" >
        <entry key="GaussianBlurScale" value="1" />
        <entry key="RemappingExponent" value="3" />
        <entry key="RemappingSteepness" value="0.04" />
        <folder key="Files" >
          <folder key="Grey" >
            <entry key="Dimensions" value="1024 1024 1288" />
            <entry key="Orientation" value="RPI" />
          </folder>
        </folder>
        <folder key="IRIS" >
          <entry key="SliceViewLayerLayout" value="Stacked" />
          <folder key="BoundingBox" >
            <entry key="InterpolationMethod" value="Nearest" />
            <entry key="ResampleDimensions" value="1024 1024 1288" />
            <entry key="SeedWithCurrentSegmentation" value="0" />
            <folder key="ROIBox[0]" >
              <entry key="Index" value="0" />
              <entry key="Size" value="1024" />
            </folder>
            <folder key="ROIBox[1]" >
              <entry key="Index" value="0" />
              <entry key="Size" value="1024" />
            </folder>
            <folder key="ROIBox[2]" >
              <entry key="Index" value="0" />
              <entry key="Size" value="1288" />
            </folder>
          </folder>
          <folder key="DisplayMapping" >
            <folder key="ColorMap" >
              <entry key="Preset" value="Grayscale" />
            </folder>
            <folder key="Curve" >
              <entry key="NumberOfControlPoints" value="3" />
              <folder key="ControlPoint[0]" >
                <entry key="tValue" value="0" />
                <entry key="xValue" value="0" />
              </folder>
              <folder key="ControlPoint[1]" >
                <entry key="tValue" value="0.5" />
                <entry key="xValue" value="0.5" />
              </folder>
              <folder key="ControlPoint[2]" >
                <entry key="tValue" value="1" />
                <entry key="xValue" value="1" />
              </folder>
            </folder>
          </folder>
          <folder key="LabelState" >
            <entry key="CoverageMode" value="OverAll" />
            <entry key="DrawingLabel" value="1" />
            <entry key="OverwriteLabel" value="0" />
            <entry key="PolygonInvert" value="0" />
            <entry key="SegmentationAlpha" value="0.5" />
          </folder>
          <folder key="LabelTable" >
            <entry key="NumberOfElements" value="6" />
            <folder key="Element[0]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="255 0 0" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="1" />
              <entry key="Label" value="Label 1" />
            </folder>
            <folder key="Element[1]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="0 255 0" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="2" />
              <entry key="Label" value="Label 2" />
            </folder>
            <folder key="Element[2]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="0 0 255" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="3" />
              <entry key="Label" value="Label 3" />
            </folder>
            <folder key="Element[3]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="255 255 0" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="4" />
              <entry key="Label" value="Label 4" />
            </folder>
            <folder key="Element[4]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="0 255 255" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="5" />
              <entry key="Label" value="Label 5" />
            </folder>
            <folder key="Element[5]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="255 0 255" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="6" />
              <entry key="Label" value="Label 6" />
            </folder>
          </folder>
          <folder key="MeshOptions" >
            <entry key="DecimateFeatureAngle" value="45" />
            <entry key="DecimateMaximumError" value="0.002" />
            <entry key="DecimatePreserveTopology" value="1" />
            <entry key="DecimateTargetReduction" value="0.95" />
            <entry key="GaussianError" value="0.03" />
            <entry key="GaussianStandardDeviation" value="0.8" />
            <entry key="MeshSmoothingBoundarySmoothing" value="0" />
            <entry key="MeshSmoothingConvergence" value="0" />
            <entry key="MeshSmoothingFeatureAngle" value="45" />
            <entry key="MeshSmoothingFeatureEdgeSmoothing" value="0" />
            <entry key="MeshSmoothingIterations" value="20" />
            <entry key="MeshSmoothingRelaxationFactor" value="0.01" />
            <entry key="UseDecimation" value="0" />
            <entry key="UseGaussianSmoothing" value="1" />
            <entry key="UseMeshSmoothing" value="0" />
          </folder>
        </folder>
        <folder key="SNAP" >
          <folder key="SnakeParameters" >
            <entry key="AdvectionSpeedExponent" value="0" />
            <entry key="AdvectionWeight" value="0" />
            <entry key="AutomaticTimeStep" value="1" />
            <entry key="Clamp" value="1" />
            <entry key="CurvatureSpeedExponent" value="-1" />
            <entry key="CurvatureWeight" value="0.2" />
            <entry key="Ground" value="5" />
            <entry key="LaplacianSpeedExponent" value="0" />
            <entry key="LaplacianWeight" value="0" />
            <entry key="PropagationSpeedExponent" value="1" />
            <entry key="PropagationWeight" value="1" />
            <entry key="SnakeType" value="RegionCompetition" />
            <entry key="SolverAlgorithm" value="ParallelSparseField" />
            <entry key="TimeStepFactor" value="1" />
          </folder>
        </folder>
      </folder>
    </folder>
"""


#######################################################################


itksnapstr = """<?xml version="1.0" encoding="UTF-8" ?>
<!--ITK-SNAP (itksnap.org) Project File

This file can be moved/copied along with the images that it references
as long as the relative location of the images to the project file is 
the same. Do not modify the SaveLocation entry, or this will not work.
-->
<!DOCTYPE registry [
<!ELEMENT registry (entry*,folder*)>
<!ELEMENT folder (entry*,folder*)>
<!ELEMENT entry EMPTY>
<!ATTLIST folder key CDATA #REQUIRED>
<!ATTLIST entry key CDATA #REQUIRED>
<!ATTLIST entry value CDATA #REQUIRED>
]>
<registry>
  <entry key="SaveLocation" value="/home/king/Data/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Known_Good_Lymphoma_Segs_Feb27" />
  <entry key="Version" value="20230925" />
  <folder key="Annotations" >
    <entry key="Format" value="ITK-SNAP Annotation File" />
    <entry key="FormatDate" value="20150624" />
  </folder>
  <folder key="Layers" >
    <folder key="Layer[000]" >
      <entry key="AbsolutePath" value="T1.nii.gz" />
      <entry key="Role" value="MainRole" />
      <entry key="Tags" value="" />
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="255" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="0" />
        <entry key="Tags" value="" />
        <folder key="DisplayMapping" >
          <folder key="ColorMap" >
            <entry key="Preset" value="Grayscale" />
          </folder>
          <folder key="Curve" >
            <entry key="NumberOfControlPoints" value="3" />
            <folder key="ControlPoint[0]" >
              <entry key="tValue" value="0" />
              <entry key="xValue" value="0" />
            </folder>
            <folder key="ControlPoint[1]" >
              <entry key="tValue" value="0.5" />
              <entry key="xValue" value="0.5" />
            </folder>
            <folder key="ControlPoint[2]" >
              <entry key="tValue" value="1" />
              <entry key="xValue" value="1" />
            </folder>
          </folder>
        </folder>
      </folder>
      <folder key="ProjectMetaData" >
        <entry key="GaussianBlurScale" value="1" />
        <entry key="RemappingExponent" value="3" />
        <entry key="RemappingSteepness" value="0.04" />
        <folder key="Files" >
          <folder key="Grey" >
            <entry key="Dimensions" value="1024 1024 1288" />
            <entry key="Orientation" value="RPI" />
          </folder>
        </folder>
        <folder key="IRIS" >
          <entry key="SliceViewLayerLayout" value="Stacked" />
          <folder key="DisplayMapping" >
            <folder key="ColorMap" >
              <entry key="Preset" value="Grayscale" />
            </folder>
            <folder key="Curve" >
              <entry key="NumberOfControlPoints" value="3" />
              <folder key="ControlPoint[0]" >
                <entry key="tValue" value="0" />
                <entry key="xValue" value="0" />
              </folder>
              <folder key="ControlPoint[1]" >
                <entry key="tValue" value="0.5" />
                <entry key="xValue" value="0.5" />
              </folder>
              <folder key="ControlPoint[2]" >
                <entry key="tValue" value="1" />
                <entry key="xValue" value="1" />
              </folder>
            </folder>
          </folder>
          <folder key="LabelState" >
            <entry key="CoverageMode" value="OverAll" />
            <entry key="DrawingLabel" value="1" />
            <entry key="OverwriteLabel" value="0" />
            <entry key="PolygonInvert" value="0" />
            <entry key="SegmentationAlpha" value="0.5" />
          </folder>
          <folder key="LabelTable" >
            <entry key="NumberOfElements" value="6" />
            <folder key="Element[0]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="255 0 0" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="1" />
              <entry key="Label" value="Label 1" />
            </folder>
            <folder key="Element[1]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="0 255 0" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="2" />
              <entry key="Label" value="Label 2" />
            </folder>
            <folder key="Element[2]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="0 0 255" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="3" />
              <entry key="Label" value="Label 3" />
            </folder>
            <folder key="Element[3]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="255 255 0" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="4" />
              <entry key="Label" value="Label 4" />
            </folder>
            <folder key="Element[4]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="0 255 255" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="5" />
              <entry key="Label" value="Label 5" />
            </folder>
            <folder key="Element[5]" >
              <entry key="Alpha" value="255" />
              <entry key="Color" value="255 0 255" />
              <entry key="Flags" value="1 1" />
              <entry key="Index" value="6" />
              <entry key="Label" value="Label 6" />
            </folder>
          </folder>
          <folder key="MeshOptions" >
            <entry key="DecimateFeatureAngle" value="45" />
            <entry key="DecimateMaximumError" value="0.002" />
            <entry key="DecimatePreserveTopology" value="1" />
            <entry key="DecimateTargetReduction" value="0.95" />
            <entry key="GaussianError" value="0.03" />
            <entry key="GaussianStandardDeviation" value="0.8" />
            <entry key="MeshSmoothingBoundarySmoothing" value="0" />
            <entry key="MeshSmoothingConvergence" value="0" />
            <entry key="MeshSmoothingFeatureAngle" value="45" />
            <entry key="MeshSmoothingFeatureEdgeSmoothing" value="0" />
            <entry key="MeshSmoothingIterations" value="20" />
            <entry key="MeshSmoothingRelaxationFactor" value="0.01" />
            <entry key="UseDecimation" value="0" />
            <entry key="UseGaussianSmoothing" value="1" />
            <entry key="UseMeshSmoothing" value="0" />
          </folder>
        </folder>
        <folder key="SNAP" >
          <folder key="SnakeParameters" >
            <entry key="AdvectionSpeedExponent" value="0" />
            <entry key="AdvectionWeight" value="0" />
            <entry key="AutomaticTimeStep" value="1" />
            <entry key="Clamp" value="1" />
            <entry key="CurvatureSpeedExponent" value="-1" />
            <entry key="CurvatureWeight" value="0.2" />
            <entry key="Ground" value="5" />
            <entry key="LaplacianSpeedExponent" value="0" />
            <entry key="LaplacianWeight" value="0" />
            <entry key="PropagationSpeedExponent" value="1" />
            <entry key="PropagationWeight" value="1" />
            <entry key="SnakeType" value="RegionCompetition" />
            <entry key="SolverAlgorithm" value="ParallelSparseField" />
            <entry key="TimeStepFactor" value="1" />
          </folder>
        </folder>
      </folder>
    </folder>
"""
dwi1str="""<folder key="Layer[000]" >
      <entry key="AbsolutePath" value="DWI1.nii.gz" />
      <entry key="Role" value="OverlayRole" />
      <entry key="Tags" value="" />
      <folder key="ImageTransform" >
        <entry key="IsIdentity" value="1" />
      </folder>
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="0.5" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="1" />
        <entry key="Tags" value="" />
        <folder key="DisplayMapping" >
          <folder key="ColorMap" >
            <entry key="Preset" value="Summer" />
          </folder>
          <folder key="Curve" >
            <entry key="NumberOfControlPoints" value="3" />
            <folder key="ControlPoint[0]" >
              <entry key="tValue" value="0" />
              <entry key="xValue" value="0" />
            </folder>
            <folder key="ControlPoint[1]" >
              <entry key="tValue" value="0.0298649" />
              <entry key="xValue" value="0.445455" />
            </folder>
            <folder key="ControlPoint[2]" >
              <entry key="tValue" value="0.266667" />
              <entry key="xValue" value="1" />
            </folder>
          </folder>
        </folder>
      </folder>
    </folder>
"""
petstr="""<folder key="Layer[000]" >
      <entry key="AbsolutePath" value="PET.nii.gz" />
      <entry key="Role" value="OverlayRole" />
      <entry key="Tags" value="" />
      <folder key="ImageTransform" >
        <entry key="IsIdentity" value="1" />
      </folder>
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="0.5" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="1" />
        <entry key="Tags" value="" />
        <folder key="DisplayMapping" >
          <folder key="ColorMap" >
            <entry key="Preset" value="Black to green" />
          </folder>
          <folder key="Curve" >
            <entry key="NumberOfControlPoints" value="3" />
            <folder key="ControlPoint[0]" >
              <entry key="tValue" value="0" />
              <entry key="xValue" value="0" />
            </folder>
            <folder key="ControlPoint[1]" >
              <entry key="tValue" value="0.0666667" />
              <entry key="xValue" value="0.5" />
            </folder>
            <folder key="ControlPoint[2]" >
              <entry key="tValue" value="0.133333" />
              <entry key="xValue" value="1" />
            </folder>
          </folder>
        </folder>
      </folder>
    </folder>
"""
dwi2str="""<folder key="Layer[000]" >
      <entry key="AbsolutePath" value="DWI2.nii.gz" />
      <entry key="Role" value="OverlayRole" />
      <entry key="Tags" value="" />
      <folder key="ImageTransform" >
        <entry key="IsIdentity" value="1" />
      </folder>
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="0.5" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="1" />
        <entry key="Tags" value="" />
        <folder key="DisplayMapping" >
          <folder key="ColorMap" >
            <entry key="Preset" value="Autumn" />
          </folder>
          <folder key="Curve" >
            <entry key="NumberOfControlPoints" value="3" />
            <folder key="ControlPoint[0]" >
              <entry key="tValue" value="0" />
              <entry key="xValue" value="0" />
            </folder>
            <folder key="ControlPoint[1]" >
              <entry key="tValue" value="0.0666667" />
              <entry key="xValue" value="0.5" />
            </folder>
            <folder key="ControlPoint[2]" >
              <entry key="tValue" value="0.133333" />
              <entry key="xValue" value="1" />
            </folder>
          </folder>
        </folder>
      </folder>
    </folder>
"""
t2str="""<folder key="Layer[000]" >
      <entry key="AbsolutePath" value="T2.nii.gz" />
      <entry key="Role" value="OverlayRole" />
      <entry key="Tags" value="" />
      <folder key="ImageTransform" >
        <entry key="IsIdentity" value="1" />
      </folder>
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="0.5" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="1" />
        <entry key="Tags" value="" />
        <folder key="DisplayMapping" >
          <folder key="ColorMap" >
            <entry key="Preset" value="Jet" />
          </folder>
          <folder key="Curve" >
            <entry key="NumberOfControlPoints" value="3" />
            <folder key="ControlPoint[0]" >
              <entry key="tValue" value="0.4" />
              <entry key="xValue" value="0" />
            </folder>
            <folder key="ControlPoint[1]" >
              <entry key="tValue" value="0.566667" />
              <entry key="xValue" value="0.5" />
            </folder>
            <folder key="ControlPoint[2]" >
              <entry key="tValue" value="0.733333" />
              <entry key="xValue" value="1" />
            </folder>
          </folder>
        </folder>
      </folder>
    </folder>
    """
suvstr="""<folder key="Layer[000]" >
      <entry key="AbsolutePath" value="SUV.nii.gz" />
      <entry key="Role" value="OverlayRole" />
      <entry key="Tags" value="" />
      <folder key="ImageTransform" >
        <entry key="IsIdentity" value="1" />
      </folder>
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="0.5" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="1" />
        <entry key="Tags" value="" />
        <folder key="DisplayMapping" >
          <folder key="ColorMap" >
            <entry key="Preset" value="Hot" />
          </folder>
          <folder key="Curve" >
            <entry key="NumberOfControlPoints" value="3" />
            <folder key="ControlPoint[0]" >
              <entry key="tValue" value="0" />
              <entry key="xValue" value="0" />
            </folder>
            <folder key="ControlPoint[1]" >
              <entry key="tValue" value="0.0666667" />
              <entry key="xValue" value="0.5" />
            </folder>
            <folder key="ControlPoint[2]" >
              <entry key="tValue" value="0.133333" />
              <entry key="xValue" value="1" />
            </folder>
          </folder>
        </folder>
      </folder>
    </folder>
"""
segstr="""<folder key="Layer[000]" >
      <entry key="AbsolutePath" value="SEG.nii.gz" />
      <entry key="Role" value="SegmentationRole" />
      <entry key="Tags" value="" />
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="4.94066e-324" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="1" />
        <entry key="Tags" value="" />
      </folder>
    </folder>
  """
itksnapstrend="""</folder>
  <folder key="TimePointProperties" >
    <entry key="FormatVersion" value="1" />
    <folder key="TimePoints" >
      <entry key="ArraySize" value="1" />
      <folder key="TimePoint[1]" >
        <entry key="Nickname" value="" />
        <entry key="Tags" value="" />
        <entry key="TimePoint" value="1" />
      </folder>
    </folder>
  </folder>
</registry>
"""

def getitkworkspace(path_to_data,d):
    
    casefiles = {"T1.nii.gz":True}
    itkcmdopts = f'-g {str(d/"T1.nii.gz")} '             #main image is T1. assumed to exist
    #dims = get_nifti_shape(d/"T1.nii.gz")#get dimensions. have to replace dim string in the itkproj
    itkproj = itksnapstr.replace("T1.nii.gz",str(d/"T1.nii.gz"))
    if (d/"SEG.nii.gz").exists():
        itkcmdopts = itkcmdopts + f'-s {str(d/"SEG.nii.gz")} '     #set segmentation file option (not assumed to exist)
        casefiles["SEG.nii.gz"] = True
        itkproj = itkproj + segstr.replace("SEG.nii.gz",str(d/"SEG.nii.gz")).replace("000",f'{len(casefiles)-1:03d}')
    for f in d.glob('*nii.gz'):
        if f.name == 'T1.nii.gz' or f.name == 'SEG.nii.gz':
            continue
        else:
            if "res" in f.name:
                continue
            if len(casefiles) == 1 or (len(casefiles) == 2 and "SEG.nii.gz" in casefiles):
                itkcmdopts = itkcmdopts + '-o '
            casefiles[f.name] =  True
            itkcmdopts = itkcmdopts + str(f)+" "
            for m in itksnapmods:
                if f.name in m:
                    itkproj = itkproj + m.replace(f.name,str(f)).replace("000",f'{len(casefiles)-1:03d}')
                    break
    itkproj = itkproj + itksnapstrend
    with open(path_to_data+"/checkingcase.json","w") as f:
        f.write(itkproj)
    return casefiles

if __name__ == "__main__":
    path_to_data = plb.Path(sys.argv[1])
    path_to_data = "/home/king/Data/Lymphoma_Data_Cache/nnUNet_raw_data_base/nnUNet_raw_data/Known_Good_Lymphoma_Segs_Feb27"
    print(f"DEBUG, hard coded data path as: {path_to_data}")

    itksnapcmd = "/opt/itksnap/itksnap-4.0.2-20230925-Linux-gcc64/bin/itksnap"
    itksnapmods = [dwi1str,dwi2str,suvstr,t2str,petstr]

    nifti_dirs = find_nifti_folders(path_to_data)
    nifti_dirs = (sorted(nifti_dirs, key=natural_sort_key))

    blacklist = []
    report = {}
    #try to avoid re-doing checked cases
    reportf = plb.Path(path_to_data)/'report.json'
    if reportf.exists():
        with open(reportf) as f:
            report = json.load(f)
        blacklist=list(report.keys())
    
    
    candidate_dirs = []
    for s in nifti_dirs:
        if not any([b in str(s) for b in blacklist]):
            candidate_dirs.append(s)

    nifti_dirs = candidate_dirs

    #high res first, low res later
    #for d in nifti_dirs:
    I = len(nifti_dirs)
    i = 0
    while i < I:
        d = nifti_dirs[i]
        casename = d.parts[-1]
        casefiles = getitkworkspace(path_to_data,d)

        def position_window(root, width=300, height=200, xf = 1.0, yf = 0.5):
    
            # get screen width and height
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()

            # calculate position x and y coordinates
            x = (screen_width*xf) - (width/2)
            y = (screen_height*yf) - (height/2)

            if x > screen_width-width:
                x  = screen_width-width
            if x < width:
                x = 0.0
            if y > screen_height-height:
                y  = screen_height-height*1.25
            if y < 0:
                y = 0
            #root.geometry('%dx%d+%d+%d' % (width, height, x, y))
            root.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
        #i = i +1
        #break        
        #load up a case
        #cmd = f'{itksnapcmd} {itkcmdopts}'
        cmd = f'{itksnapcmd} -w {path_to_data+"/checkingcase.json"}'
        p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)

        window = tk.Tk()
        window.title(f'Check: {casename} (may take time to load)')
        #window.geometry('600x500')
        position_window(window,600,500)
        
        #window.protocol("WM_DELETE_WINDOW", on_quit)

        l = tk.Label(window, bg='gray', width=20, text='empty')
        l.grid(row = 0, column = 0, sticky='n')


        #button functions
        def next_btn_press():
            #log data
            cased = {"T1": 1, "SEG":imglbl.get(), "T2":imgt2.get(), "SUV":imgsuv.get(), "DWI1":imgdwi1.get(), "DWI2":imgdwi2.get()}
            report[casename] = cased
            p.kill()
            try:
                p.communicate()
            finally:
                print("p is dead")

            print("killed itksnap")
            window.destroy()        #kill the main loop

        def reload_btn_press(p):
            if p.returncode is None:
                p.kill()
                try:
                    p.communicate()
                finally:
                    print("p is dead (before reloading)")           
            p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True) #
            print("reload case")

        def prev_btn_press(p,i):
            if p.returncode is None:
                p.kill()
                try:
                    p.communicate()
                finally:
                    print("p is dead (before reloading)") 
            if i > 0:
                i = i-1

            window.destroy()
            print("reload case") 

        #forward reset backward buttons
        button1=tk.Button(window, text="PREV", command=lambda: prev_btn_press(p,i))
        nxtbtn=tk.Button(window, text="NEXT", command=next_btn_press)
        rldbtn=tk.Button(window, text="RELOAD", command=lambda: reload_btn_press(p))
        button1.grid(row = 1, column = 0,  pady = 1)
        nxtbtn.grid(row = 1, column = 1,  pady = 1)
        rldbtn.grid(row = 1, column = 2,  pady = 1)
        #close button override
        def on_close():
            save_report(path_to_data+'/report.json',report)
            window.destroy()
            exit()

        window.protocol("WM_DELETE_WINDOW",  on_close)

        def print_selection():
            pass
            if (imglbl.get() == 1) & (imgsuv.get() == 0):
                l.config(text='I love Python ')
            elif (imglbl.get() == 0) & (imgsuv.get() == 1):
                l.config(text='I love C++')
            elif (imglbl.get() == 0) & (imgsuv.get() == 0):
                l.config(text='I do not anything')
            else:
                l.config(text='I love both')
        
        imglbl = tk.IntVar()
        imglbl.set(-1)
        imgsuv = tk.IntVar()
        imgsuv.set(-1)
        imgt2 = tk.IntVar()
        imgt2.set(-1)
        imgdwi1 = tk.IntVar()
        imgdwi1.set(-1)
        imgdwi2 = tk.IntVar()
        imgdwi2.set(-1)
        imgpet = tk.IntVar()
        imgpet.set(-1)
        if "SEG.nii.gz" in casefiles:
            imglbl = tk.IntVar()
            c1 = tk.Checkbutton(window, text='Labels OK?',variable=imglbl, onvalue=1, offvalue=0, command=print_selection, foreground='black')
            c1.select()
            c1.grid(row = 2, column = 1, sticky='W', pady = 2)
            

        if "SUV.nii.gz" in casefiles:
            imgsuv = tk.IntVar()
            c2 = tk.Checkbutton(window, text='SUV OK?',variable=imgsuv, onvalue=1, offvalue=0, command=print_selection, foreground='black')
            c2.select()
            c2.grid(row = 3, column = 1,  sticky='W', pady = 2)

        if "T2.nii.gz" in casefiles:
            imgt2 = tk.IntVar()
            c3 = tk.Checkbutton(window, text='T2 OK?',variable=imgt2, onvalue=1, offvalue=0, command=print_selection, foreground='black')
            c3.select()
            c3.grid(row = 4, column = 1,  sticky='W', pady = 2)
        if "DWI2.nii.gz" in casefiles:
            imgdwi2 = tk.IntVar()
            c4 = tk.Checkbutton(window, text='DWI2 OK?',variable=imgdwi2, onvalue=1, offvalue=0, command=print_selection, foreground='black')
            c4.select()
            c4.grid(row = 5, column = 1,  sticky='W', pady = 2)
        if "DWI1.nii.gz" in casefiles:
            imgdwi1 = tk.IntVar()
            c5 = tk.Checkbutton(window, text='DWI1 OK?',variable=imgdwi1, onvalue=1, offvalue=0, command=print_selection, foreground='black')
            c5.select()
            c5.grid(row = 6, column = 1,  sticky='W', pady = 2)

        if "PET.nii.gz" in casefiles:
            imgpet = tk.IntVar()
            c6 = tk.Checkbutton(window, text='PET OK?',variable=imgpet, onvalue=1, offvalue=0, command=print_selection, foreground='black')
            c6.select()
            c6.grid(row = 7, column = 1,  sticky='W', pady = 2)

        window.mainloop()

        print("write results")
        save_report(path_to_data+'/report.json',report)
        del(p)
        i = i +1
   