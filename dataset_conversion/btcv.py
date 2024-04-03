import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground
import os
import random
import yaml
import copy
import pdb

def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    imLabel.CopyInformation(imImage)

    assert imImage.GetSize() == imLabel.GetSize()


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()
    

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))

    imImage.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    imLabel.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)
    
    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)


    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[10, 30, 30])

    sitk.WriteImage(cropped_img, '%s/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz'%(save_path, name))


def load_from_file(path):
    """Function to load the data from a file.

    Args:
        path (str): Path to the file.
    """
            
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), path))  
    return path


if __name__ == '__main__':


    src_path = load_from_file('../../Datasets/BTCV_')
    tgt_path = load_from_file('../../Datasets/BTCV2_')

    name_list = os.listdir(src_path + '/imagesTr')
    name_list = [name.split('.')[0] for name in name_list]

    if not os.path.exists(tgt_path+'/list'):
        os.mkdir('%s/list'%(tgt_path))
    with open("%s/list/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)

    for name in name_list:
        img_name = name + '.nii.gz'
        lab_name = img_name.replace('img', 'label')

        img = sitk.ReadImage(src_path+'/imagesTr/%s'%img_name)
        lab = sitk.ReadImage(src_path+'/labelsTr/%s'%lab_name)

        ResampleImage(img, lab, tgt_path, name, (1.5, 1.5, 2.0)) #  (0.767578125, 0.767578125, 1.0)  ||| (1.5, 1.5, 2.0)
        print(img_name, 'done')
        print(lab_name, 'done')



