# coding=utf-8
import numpy as np
import transforms3d
import random
import math
from PIL import Image
import glob
import os
import shutil
import re

def augment_cloud(Ps, args, return_augmentation_params=False):
    """" Augmentation on XYZ and jittering of everything """
    # Ps is a list of point clouds

    M = transforms3d.zooms.zfdir2mat(1) # M is 3*3 identity matrix
    # scale
    if args['pc_augm_scale'] > 1:
        s = random.uniform(1/args['pc_augm_scale'], args['pc_augm_scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)

    # rotation
    if args['pc_augm_rot']:
        scale = args['pc_rot_scale'] # we assume the scale is given in degrees
        # should range from 0 to 180
        if scale > 0:
            angle = random.uniform(-math.pi, math.pi) * scale / 180.0
            M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) # y=upright assumption
            # we have verified that shapes from mvp data, the upright direction is along the y axis positive direction
    
    # mirror
    if args['pc_augm_mirror_prob'] > 0: # mirroring x&z, not y
        if random.random() < args['pc_augm_mirror_prob']/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < args['pc_augm_mirror_prob']/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)

    # translation
    translation_sigma = args.get('translation_magnitude', 0)
    translation_sigma = max(args['pc_augm_scale'], 1) * translation_sigma
    if translation_sigma > 0:
        noise = np.random.normal(scale=translation_sigma, size=(1, 3))
        noise = noise.astype(Ps[0].dtype)
        
    result = []
    for P in Ps:
        P[:,:3] = np.dot(P[:,:3], M.T)
        if translation_sigma > 0:
            P[:,:3] = P[:,:3] + noise
        if args['pc_augm_jitter']:
            sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
            P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
        result.append(P)

    if return_augmentation_params:
        augmentation_params = {}
        augmentation_params['M_inv'] = np.linalg.inv(M.T).astype(Ps[0].dtype)
        if translation_sigma > 0:
            augmentation_params['translation'] = noise
        else:
            augmentation_params['translation'] = np.zeros((1, 3)).astype(Ps[0].dtype)
        
        return result, augmentation_params

    return result

def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) if len(c) > 0]
    return sorted(file_list_ordered, key=alphanum_key)

def get_file_num(file_path):
    file_num = glob.glob(file_path)
    return len(file_num)

def scanNet_check(scan_path,remove=False):

    scans = [name.split("/")[-1] for name in glob.glob(scan_path+"/scene*")]
    scans = sorted_alphanum(scans)
    invalid_folders = []
    valid_folders = []
    for i,scan_name in enumerate(scans):
        # print(f"-------------------------- i:{i}, {scan_name} ----------------------------")
        folder_path = os.path.join(scan_path,scan_name)
        folders = glob.glob(os.path.join(folder_path,"*"))
        folders_num = len(folders)
        if(folders_num<4):
            print(f"---- Invalid folder : {scan_name}, folder_num : {folders_num}, foloder : {folders}----")
            invalid_folders.append(scan_name)
            if(remove):
                shutil.rmtree(folder_path)
                print(f"Removing : {folder_path}")
            continue
        colors_path = os.path.join(scan_path,scan_name,"color/*.png")
        colors_num = get_file_num(colors_path)
        depths_path = os.path.join(scan_path,scan_name,"depth/*.png")
        depths_num = get_file_num(depths_path)
        poses_path = os.path.join(scan_path,scan_name,"pose/*.txt")
        poses_num = get_file_num(poses_path)
        intrinsic_path = os.path.join(scan_path,scan_name,"intrinsic/*.txt")
        intrinsic_num = get_file_num(intrinsic_path)
        if(colors_num != depths_num or colors_num != poses_num or intrinsic_num != 4):
            print(f"---- Invalid folder : {scan_name}, color : {colors_num}, depth : {depths_num}, pose : {poses_num}, intrinsic_num : {intrinsic_num} ----")
            invalid_folders.append(scan_name)
            if(remove):
                shutil.rmtree(folder_path)
                print(f"Removing : {folder_path}")
            continue
        valid_folders.append(scan_name)
        # print(f"|||| Valid folder : {scan_name}, folder : {folders} ||||")
        # print(f"-------------------------- i:{i}, {scan_name} ----------------------------\n")
    return valid_folders

def scanNet_origin_check(scan_path,mode="test",remove=False):

    scans = [name.split("/")[-1] for name in glob.glob(scan_path+"/scene*")]
    scans_num = len(scans)
    print(f"Current ScanNet Scene Number : {scans_num}, Mode : {mode}")
    scans = sorted_alphanum(scans)
    invalid_folders = []
    '''
        train:
            scene0185_00 -- scene0185_00_2d-instance-filt.zip not exist!
            scene0185_00 -- scene0185_00_2d-label.zip not exist!
            scene0185_00 -- scene0185_00_2d-label-filt.zip not exist!
            //// scene0185_00 is not Safe! \\\\
        test:
            Invalid Scenes Num : 0
    '''
    for i,scan_name in enumerate(scans):
        if(mode == "train"):
            files = [
                f"{scan_name}.aggregation.json",
                f"{scan_name}.sens",
                f"{scan_name}_vh_clean.aggregation.json",
                f"{scan_name}_vh_clean.ply",
                f"{scan_name}_vh_clean.segs.json",
                f"{scan_name}_vh_clean_2.0.010000.segs.json",
                f"{scan_name}_vh_clean_2.labels.ply",
                f"{scan_name}_vh_clean_2.ply",
                f"{scan_name}_2d-instance.zip",
                f"{scan_name}_2d-instance-filt.zip",
                f"{scan_name}_2d-label.zip",
                f"{scan_name}_2d-label-filt.zip",
                f"{scan_name}.txt",
            ]
        elif(mode == "test"):
            files = [
                f"{scan_name}.sens",
                f"{scan_name}_vh_clean.ply",
                f"{scan_name}_vh_clean_2.ply",
                f"{scan_name}.txt",
            ]
        else:
            print("Error Mode !")
            exit(1)
        fige = 0
        for file in files:
            file_path = os.path.join(scan_path,scan_name,file)
            if(not os.path.exists(file_path)):
                print(f"{scan_name} -- {file} not exist!")
                fige = 1
        if(fige):
            invalid_folders.append(scan_name)
            print(f"//// {scan_name} is not Safe! \\\\\\\\")
        else:
            print(f"---- {scan_name} is Safe! ----")

    print(f"Invalid Scenes Num : {len(invalid_folders)}")
    for folder in invalid_folders:
        print(f"Invalid Scene : {folder}")
        if(remove):
            invalid_scan_path = os.path.join(scan_path,folder)
            shutil.rmtree(invalid_scan_path)
            print(f"Removing Scene: {folder}")

def threeMatch_analysis(analysis_path = "/DISK/qwt/datasets/3dmatch/analysis_png"):

    folders = [name.split("/")[-1] for name in glob.glob(analysis_path+"/*")]
    for folder in folders:
        seq_path = os.path.join(analysis_path,folder,"seq*")
        seqs = [name.split("/")[-1] for name in glob.glob(seq_path)]
        for seq in seqs:
            image_path = os.path.join(analysis_path,folder,seq,"*.color.*")
            images = [name.split("/")[-1].split(".")[0] for name in glob.glob(image_path)]
            for image_name in images:
                jpg_path = os.path.join(analysis_path,folder,seq,f"{image_name}.color.jpg")
                image = Image.open(jpg_path)
                # image = uio.process_image(image)
                image = image.resize((640, 480), Image.BILINEAR)
                png_path = os.path.join(analysis_path,folder,seq,f"{image_name}.color.png")
                image.save(png_path,filename=png_path)
                print(f"Saving : {png_path}")

def get_folder_size(folder_path="/DISK/qwt"):
    '''
        getting the file size of target folder(B).
        1KB = 1024B
        1MB = 1024KB
        1GB = 1024MB
        1TB = 1024GB
    '''
    file_size = 0
    file_num = 0
    if(os.path.isdir(folder_path)):
        folders = os.listdir(folder_path)
        for folder in folders:
            folder = os.path.join(folder_path,folder)
            results = get_folder_size(folder)
            file_size += results[0]
            file_num += results[1]
    else:
        file_size += os.path.getsize(folder_path)
        file_num += 1
    return file_size, file_num

if __name__ == '__main__':
    import pdb
    args = {'pc_augm_scale':0, 'pc_augm_rot':False, 'pc_rot_scale':30.0, 'pc_augm_mirror_prob':0.5, 'pc_augm_jitter':False}
    N = 2048
    C = 6
    num_of_clouds = 2
    pc = []
    for _ in range(num_of_clouds):
        pc.append(np.random.rand(N,C)-0.5)
    
    result = augment_cloud(pc, args)
    pdb.set_trace()


