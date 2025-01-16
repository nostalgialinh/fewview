#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import collections
import numpy as np
import torch
from PIL import Image as pil_image
from scene.cameras import Camera
import numpy as np
from tqdm import tqdm
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import matplotlib.pyplot as plt
from skimage.transform import resize
from utils.depth_utils import estimate_depth, midas_depth
import math

WARNED = False




def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
            
        

def resize_depth_map(depth_map_mde, old_size, new_size):
    scale_factor = new_size[0] // old_size[0] 
    assert new_size[0] % old_size[0] == 0 and new_size[1] % old_size[1] == 0, \
        "New size must be an exact multiple of the old size."

    depth_map_resized = np.repeat(np.repeat(depth_map_mde, scale_factor, axis=0), scale_factor, axis=1)
    # print(np.shape(depth_map_mde))
    # print(np.shape(depth_map_resized))
    return depth_map_resized

def patchify(images):
    all_batches = []
    for image in images:
        batch = {}
        patch_size = image.img_patch_size 
        depth_pcd = image.depth_map_pcd
        depth_mde = image.depth_map_mde
        if image.key_idx is None:
            continue
        for x,y in image.key_idx:
            wid = x // patch_size
            hei = y // patch_size
            if(wid not in batch):
                batch[wid] = {}
            if(hei not in batch[wid]):
                batch[wid][hei]=[]
            
            batch[wid][hei].append({
                        "x": x,
                        "y": y,
                        "depth_pcd": depth_pcd[y][x],
                        "depth_mde": depth_mde[y][x]
            })
        all_batches.append(batch)
    return all_batches
  
def optimize_depth_batch(source, target, mask, depth_weight, patch_size=1, prune_ratio=0.001):
    """
    Optimize depth per patch (batch by patch).
    
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    patch_size: int
        Size of each patch.
    Returns
    =======
    refined_source: np.array(h,w)
        Refined source image.
    losses: list of float
        Loss values for each patch.
    """

    unskip_mask = torch.from_numpy(np.ones(source.shape)).cuda()
    source = torch.from_numpy(source).cuda()
    target = torch.from_numpy(target).cuda()
    mask = torch.from_numpy(mask).cuda()
    depth_weight = torch.from_numpy(depth_weight).cuda()

    h, w = source.shape
    refined_source = torch.zeros_like(source)
    losses = 0
    

    # Prune outlier depths
    with torch.no_grad():
        target_depth_sorted = target[target > 1e-7].sort().values
        # print(f"\n\n Love this {target_depth_sorted.size()}\n")
        min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel() * prune_ratio)]
        max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel() * (1.0 - prune_ratio))]

        mask2 = target > min_prune_threshold
        mask3 = target < max_prune_threshold
        mask = torch.logical_and(torch.logical_and(mask, mask2), mask3)

    # Iterate through patches
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # Define patch region
            patch_mask = mask[i:i + patch_size, j:j + patch_size]
            const_patch_source = source[i:i + patch_size, j:j + patch_size]
            patch_target = target[i:i + patch_size, j:j + patch_size]
            patch_depth_weight = depth_weight[i:i + patch_size, j:j + patch_size]

            # Skip empty patches
            non_zeros = torch.count_nonzero(patch_mask)
            non_zeros_percentage = non_zeros/h*w

            if non_zeros_percentage < 0.4:
                unskip_mask[i:i + patch_size, j:j + patch_size] = 0 
                continue

            # Flatten and filter valid points in the patch
            patch_source = const_patch_source[patch_mask]
            patch_target = patch_target[patch_mask]
            patch_depth_weight = patch_depth_weight[patch_mask]

            # Initialize scale and shift
            scale = torch.ones(1).cuda().requires_grad_(True)
            shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

            optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
            loss_prev = 1e6
            loss_ema = 0.0
            iteration = 1

            while True:
                patch_source_hat = scale * patch_source + shift
                loss = torch.mean(((patch_target - patch_source_hat)**2) * patch_depth_weight)

                # Penalize values outside [0,1]
                loss_hinge1 = loss_hinge2 = 0.0
                if (patch_source_hat <= 0.0).any():
                    loss_hinge1 = 2.0 * ((patch_source_hat[patch_source_hat <= 0.0])**2).mean()
                # if (patch_source_hat >= 1.0).any():
                #     loss_hinge2 = 0.3 * ((patch_source_hat[patch_source_hat >= 1.0])**2).mean()

                loss = loss + loss_hinge1 + loss_hinge2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                iteration += 1
                loss_ema = loss.item() * 0.2 + loss_ema * 0.8

                if iteration % 1000 == 0:
                    # print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
                    loss_prev = loss.item()

                if abs(loss_ema - loss_prev) < 1e-5:
                    break

            losses += loss.item()

            # Update refined source for the patch
            with torch.no_grad():
                refined_patch = scale * const_patch_source + shift
                refined_source[i:i + patch_size, j:j + patch_size] = refined_patch

    torch.cuda.empty_cache()
    # refined_source, loss_global = optimize_depth_global(refined_source.cpu().numpy(), target, mask, depth_weight)
    print('\nDone optimizing linear regression\n')
    return refined_source.cpu().numpy(), losses, unskip_mask.cpu().numpy()



def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    mask = None if cam_info.mask is None else cv2.resize(cam_info.mask, resolution)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    depth = estimate_depth(gt_image.cuda()).cpu().numpy()

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,  image=gt_image, gt_alpha_mask=loaded_mask,
                  uid=id, data_device=args.data_device, image_name=cam_info.image_name,
                  depth_image=depth, mask=mask, bounds=cam_info.bounds, depth_midas=None)

def loadCam2(args, id, cam_info, resolution_scale, n_views, pcd):
    orig_w, orig_h = cam_info.image.size
    width = cam_info.width
    height = cam_info.height
    img_size = (orig_w, orig_h)
    print(f"\nImage size {orig_w}\n\n")

    refined_depth = None
    depth = None
    depth_global = None
    
    shape = (orig_h,orig_w)
    depth_pcd = np.zeros(shape)
    depth_mde = np.zeros(shape)
    depth_weight = np.zeros(shape)
    mask = np.zeros(shape)
    # print(f"\n Mask shape {mask.shape}\n ")
    resolution = 8

    # print('Train\n')
    ply_path = os.path.join(args.source_path, str(n_views) + "_views/dense/fused.ply")
    # pcd = fetchPly(ply_path)
    
    depth_path = cam_info.rgb_path

    focal_length_x = fov2focal(cam_info.FovX, width)
    focal_length_y = fov2focal(cam_info.FovY, height)

    K = np.array([[focal_length_x, 0, width//resolution/2], [0, focal_length_y, height//resolution/2], [0, 0, 1]])
    R = cam_info.R
    T = cam_info.T

    #projection
    
    cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)) ### for coordinate definition, see getWorld2View2() function
    valid_idx = np.where(np.logical_and.reduce((cam_coord[2]>0, cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=width//resolution-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=height//resolution-1)))[0]
    pts_depths = cam_coord[-1:, valid_idx]
    cam_coord = cam_coord[:2, valid_idx]/cam_coord[-1:, valid_idx]
    depth_pcd[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = pts_depths
    depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = 1/pcd.errors[valid_idx] if pcd.errors is not None else 1
    depth_weight = depth_weight/depth_weight.max()
    

    # print(f'{cam_info.image_name}\n\n')
    depth_mde = midas_depth(depth_path)
    print(f'\nHUHUHU {depth_mde.shape}\n')
    patch_size = int(math.sqrt(orig_w*orig_h/108))
    # print(f"PATCH SICE {patch_size}\n")
    refined_depth, losses, processed_mask = optimize_depth_batch(depth_mde, depth_pcd, depth_pcd > 0.0, depth_weight, patch_size)
    # refined_depth_global, global_loss = optimize_depth_global(depth_mde, depth_pcd, depth_pcd >0.0,depth_weight)



    depth = refined_depth * processed_mask
    plt.imshow(depth)

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # print("\n Refine size \n\n")
    # print(refined_depth.size())

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    mask = None if cam_info.mask is None else cv2.resize(cam_info.mask, resolution)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None


    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,  image=gt_image, gt_alpha_mask=loaded_mask,
                  uid=id, data_device=args.data_device, image_name=cam_info.image_name,
                  depth_image=depth, mask=mask, bounds=cam_info.bounds, depth_midas = depth_mde, processed_mask = processed_mask)


def cameraList_from_camInfos2(cam_infos, resolution_scale, args, pcd):
    camera_list = []
    for id, c in tqdm((enumerate(cam_infos))):
        camera_list.append(loadCam2(args, id, c, resolution_scale, len(cam_infos), pcd))
    return camera_list

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in tqdm((enumerate(cam_infos))):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
