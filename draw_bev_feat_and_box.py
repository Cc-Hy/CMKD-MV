from skimage import io

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np




def featuremap_to_greymap(feature_map):
    """
    feature_map: (C, sizey, sizex)
    grey_map: (sizey, sizex)
    """
    import torch
    import numpy as np
    import cv2
    if len(feature_map.shape) == 3:
        feature_map = feature_map.unsqueeze(dim=0) # (b, c, sizey, sizex)
    elif len(feature_map.shape) == 4:
        pass
    else:
        raise NotImplementedError 
    # 1. GPA, (B, C, sizey, sizex) -> (B, C, 1, 1)
    channel_weights = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1,1))
    # 2. reweighting sum cross channels, (B, C, sizey, sizex) -> (B, sizey, sizex) -> (sizey, sizex)
    reduced_map = (channel_weights * feature_map).sum(dim=1).squeeze(dim=0)
    # 3. clamp
    reduced_map = torch.relu(reduced_map)
    # 4. normalize
    a_min = torch.min(reduced_map)
    a_max = torch.max(reduced_map)
    normed_map = (reduced_map - a_min) / (a_max - a_min)
    # 5. output
    grey_map = normed_map
    return grey_map

def greymap_to_rgbimg(map_grey, background=None, background_ratio=0.2, CHW_format=False):
    """
    map_grey: np, (sizey, sizex), values in 0-1
    background: np, (sizey, sizex, 3), values in 0-255.
    """
    import torch
    import numpy as np
    import cv2
    if background is None:
        background = np.zeros((map_grey.shape[0], map_grey.shape[1], 3))
    map_uint8 = (255 * map_grey).astype(np.uint8) # 0-255
    map_bgr = cv2.applyColorMap(map_uint8, cv2.COLORMAP_JET) # 0-255
    map_rbg = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)
    map_img = map_rbg + background_ratio * background
    map_img = np.clip(map_img, a_min=0, a_max=255).astype(np.uint8)
    if CHW_format:
        # (sizey, sizex, 3) -> (3, sizey, sizex)
        map_img = np.transpose(map_img, (2,0,1))
    return map_img

up = torch.nn.UpsamplingNearest2d(scale_factor=2)

# bev_lidar B,C,W,H

feat = up(feat[i].unsqueeze(0)).squeeze()

gray = featuremap_to_greymap(feat)    #feat:tensor(B,C,H,W)

rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())

plt.imshow(rgb)



# change
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [2, -30.08, -3.0, 46.8, 30.08, 1.0]
voxel_size = [0.04, 0.04]
feature_map_stride = 4
i = 0
feat =  bev_lidar  #B,C,W,H


gtb = gt_bboxes_3d[i]
gtc = gtb.corners   # N,8,3

for j in range(gtc.shape[0]):
    cornersbev = gtc[j, [0,2,6,4], 0:2]    # 4, 2
    cornersbev[:,0] = np.clip(cornersbev[:,0], a_min=point_cloud_range[0], a_max=point_cloud_range[3]) #(4,)
    cornersbev[:,1] = np.clip(cornersbev[:,1], a_min=point_cloud_range[1], a_max=point_cloud_range[4]) #(4,)
    corner_coor_x = (cornersbev[:,0] - point_cloud_range[0]) / voxel_size[0] / feature_map_stride
    corner_coor_y = (cornersbev[:,1] - point_cloud_range[1]) / voxel_size[1] / feature_map_stride
    corner_coor = np.stack([corner_coor_x, corner_coor_y], axis=1)
    corner_coor_int = np.around(corner_coor).astype(np.int32).reshape(-1,1,2)
    cv2.polylines(rgb,[corner_coor_int],True,(0,255,0))

plt.imshow(rgb)




# vis nuscense

from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')
sample_id = img_metas[0]['sample_idx']
sample = nusc.get('sample', sample_id)
for k,v in sample['data'].items():
    if 'CAM' in k:
        nusc.render_sample_data(nusc.get('sample_data', v)['token'])
