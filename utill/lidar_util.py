import numpy as np
import utill.math_util as mu
from utill import geom_util as gu
def lidar_to_cy(xyz,res_scale):
    norm = np.sqrt(np.sum(xyz ** 2, axis=1))
    ##
    log_norm = np.log(norm+1)
    norm_normal = norm/500
    rgb = norm_normal*(256**3 -1)
    # rgb = int(rgb)
    rgb = mu.convert_num_base(rgb,256)
    depth_val_r = rgb[:,2]
    depth_val_g = rgb[ :, 1]
    depth_val_b = rgb[:, 0]


    ##
    # normalized_depth = (norm) / np.max(norm)
    # logdepth = np.ones(normalized_depth.shape) + \
    #            (np.log(normalized_depth) / 5.70378)
    # logdepth = np.clip(logdepth, 0.0, 1.0)
    # logdepth *= 255.0
    # depth_val_r =logdepth
    # depth_val_g =logdepth
    # depth_val_b= logdepth
    # depth_val = 32 * np.log2(depth_val + 1)
    thetaz = np.rad2deg(np.arcsin(xyz[:, 2] / norm))
    thetax = (180 * (xyz[:, 0] < 0)) + np.rad2deg(np.arctan(xyz[:, 1] / xyz[:, 0]))
    theta_x = (thetax + 90 - 180) % 360
    theta_z = thetaz + 90
    img_cy = np.zeros((180 * res_scale, 360 * res_scale, 3))
    img_cy[np.int64(theta_z * res_scale), np.int64(theta_x * res_scale),0] = (depth_val_r)
    img_cy[np.int64(theta_z * res_scale), np.int64(theta_x * res_scale), 1] =(depth_val_g)
    img_cy[np.int64(theta_z * res_scale), np.int64(theta_x * res_scale), 2] =(depth_val_b)
    # img_cy = 255*(img_cy - np.min(img_cy))/np.max(img_cy)
    return img_cy


def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX'];
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY'];
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ'];
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
                PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]

    PointCloud[:, 2] = PointCloud[:, 2]
    return PointCloud
def BVtoPC(img,img_size,boundry_condition):
    img_size = np.float(img_size)
    x,y = img.sum(-1).nonzero()
    x_rw = ((x.reshape([-1,1])/img_size) -.5)*2*boundry_condition
    y_rw = ((y.reshape([-1, 1]) / img_size) - .5) * 2 * boundry_condition
    z = np.zeros_like(x_rw)
    vals = img[x,y]
    pc =np.concatenate([x_rw,y_rw,z],axis=-1)
    return pc,vals
def PCtoBV(pc,img_size,boundry_condition,vals):
    # img = makeBVFeature(pc,2*boundry_condition/img_size)
    Discretization = 2*np.float(boundry_condition)/np.float(img_size)
    Height = Width =img_size+1
    PointCloud = np.copy(pc)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization) + Height / 2)
    # PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)
    PointCloud[:, 2] = -PointCloud[:, 2]
    indices = np.lexsort((PointCloud[:, 2] / 100, PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]
    # Height Map
    heightMap = np.zeros((Height, Width,vals.shape[1]))

    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = vals[indices]
    return  heightMap[0:1024, 0:1024, :]
def BV2BV(img,img_size,boundry_conditions,from_mdy,to_mdy):
    pc_coop_coop_coord, vals = BVtoPC(img, img_size, boundry_conditions['maxX'])
    pc_coop_ego_coord = gu.world_to_ego_sensor(gu.ego_to_world_sensor(pc_coop_coop_coord.transpose(), from_mdy),
                                               to_mdy).transpose()
    pc_coop_ego_coord_crop = removePoints(pc_coop_ego_coord, boundry_conditions)
    img_coop_ego_coord2 = PCtoBV(pc_coop_ego_coord_crop, img_size, boundry_conditions['maxX'], vals)
    return img_coop_ego_coord2
def makeBVFeature(PointCloud_, BoundaryCond, Discretization):
    # 1024 x 1024 x 3
    import open3d as od
    Height = 1024 + 1
    Width = 1024 + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization)+Height/2)
    # PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization)+Width/2)
    PointCloud[:,2] = PointCloud[:, 2]
    # pcd = od.PointCloud()
    # pcd.points = od.Vector3dVector(PointCloud)
    # od.draw_geometries([pcd])
    # sort-3times
    indices = np.lexsort((PointCloud[:, 2]/100, PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))
    height_inverse_Map = np.zeros((Height, Width))

    # _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    # PointCloud_frac = PointCloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)


    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    # intensityMap[500:524,500:524]=1
    densityMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:3], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]
    # _, indices, counts = np.unique(PointCloud[:, 0:3], axis=0, return_index=True, return_counts=True,return_inverse=True)
    # PointCloud_bottom = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2]
    height_inverse_Map[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_bottom[:, 2]
    # heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = 1
    # intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts
    """
    plt.imshow(densityMap[:,:])
    plt.pause(2)
    plt.close()
    plt.show()
    plt.pause(2)
    plt.close()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    plt.imshow(intensityMap[:,:])
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    """
    RGB_Map = np.zeros((Height, Width, 3))
    RGB_Map[:, :, 0] = densityMap  # r_map
    RGB_Map[:, :, 1] = heightMap  # g_map
    RGB_Map[:, :, 2] = height_inverse_Map  # b_map

    # save = np.zeros((512,1024,3))
    save = RGB_Map[0:1024, 0:1024, :]
    # TODO EHSAN EDIT
    # save = RGB_Map[0:1024, 0:1024, :]
    # misc.imsave('test_bv.png',save[::-1,::-1,:])
    # misc.imsave('test_bv.png',save)
    return save
def makeBVFeature_multi_channel(PointCloud_, BoundaryCond, Discretization,channels=3.0):
    # 1024 x 1024 x 3
    import open3d as od
    Height = 1024 + 1
    Width = 1024 + 1
    delta_z = BoundaryCond['maxZ']-BoundaryCond['minZ']+1e-1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization)+Height/2)
    # PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization)+Width/2)
    PointCloud[:,2] = np.int_(np.floor(((PointCloud[:, 2]-BoundaryCond['minZ'])*channels/delta_z)))
    # pcd = od.PointCloud()
    # pcd.points = od.Vector3dVector(PointCloud)
    # od.draw_geometries([pcd])
    # sort-3times
    indices = np.lexsort((PointCloud[:, 2]/100, PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width,np.int(channels)))
    # height_inverse_Map = np.zeros((Height, Width))

    # _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    # PointCloud_frac = PointCloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)

    #
    # # Intensity Map & DensityMap
    # intensityMap = np.zeros((Height, Width))
    # # intensityMap[500:524,500:524]=1
    # densityMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:3], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]
    # _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True,return_inverse=True)
    # PointCloud_bottom = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1]),np.int_(PointCloud_top[:, 2])] = normalizedCounts
    # height_inverse_Map[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_bottom[:, 2]
    # # heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = 1
    # # intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2]
    # densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts
    """
    plt.imshow(densityMap[:,:])
    plt.pause(2)
    plt.close()
    plt.show()
    plt.pause(2)
    plt.close()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    plt.imshow(intensityMap[:,:])
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    """
    # RGB_Map = np.zeros((Height, Width, 3))
    # RGB_Map[:, :, 0] = densityMap  # r_map
    # RGB_Map[:, :, 1] = heightMap  # g_map
    # RGB_Map[:, :, 2] = height_inverse_Map  # b_map
    #
    # # save = np.zeros((512,1024,3))
    # save = RGB_Map[0:1024, 0:1024, :]
    # TODO EHSAN EDIT
    # save = RGB_Map[0:1024, 0:1024, :]
    # misc.imsave('test_bv.png',save[::-1,::-1,:])
    # misc.imsave('test_bv.png',save)
    return heightMap[:1024,:1024,:]
def lidar_to_BEV(xyz,res_scale):
    norm = np.sqrt(np.sum(xyz ** 2, axis=1))
    ##
    log_norm = np.log(norm+1)
    norm_normal = norm/500
    rgb = norm_normal*(256**3 -1)
    # rgb = int(rgb)
    rgb = mu.convert_num_base(rgb,256)
    depth_val_r = rgb[:,2]
    depth_val_g = rgb[ :, 1]
    depth_val_b = rgb[:, 0]


    ##
    # normalized_depth = (norm) / np.max(norm)
    # logdepth = np.ones(normalized_depth.shape) + \
    #            (np.log(normalized_depth) / 5.70378)
    # logdepth = np.clip(logdepth, 0.0, 1.0)
    # logdepth *= 255.0
    # depth_val_r =logdepth
    # depth_val_g =logdepth
    # depth_val_b= logdepth
    # depth_val = 32 * np.log2(depth_val + 1)
    thetaz = np.rad2deg(np.arcsin(xyz[:, 2] / norm))
    thetax = (180 * (xyz[:, 0] < 0)) + np.rad2deg(np.arctan(xyz[:, 1] / xyz[:, 0]))
    theta_x = (thetax + 90 - 180) % 360
    theta_z = thetaz + 90
    img_cy = np.zeros((180 * res_scale, 360 * res_scale, 3))
    img_cy[np.int64(theta_z * res_scale), np.int64(theta_x * res_scale),0] = (depth_val_r)
    img_cy[np.int64(theta_z * res_scale), np.int64(theta_x * res_scale), 1] =(depth_val_g)
    img_cy[np.int64(theta_z * res_scale), np.int64(theta_x * res_scale), 2] =(depth_val_b)
    # img_cy = 255*(img_cy - np.min(img_cy))/np.max(img_cy)
    return img_cy
def lidar_to_world(xyz,sen_mdy):
    return NotImplemented
def lidar_points_on_cy_image(xyz,res_scale):
    norm = np.sqrt(np.sum(xyz ** 2, axis=1))
    depth_val = 255 * (norm) / 120
    depth_val = 32 * np.log2(depth_val + 1)
    thetaz = np.rad2deg(np.arcsin(xyz[:, 2] / norm))
    thetax = (180 * (np.int64(xyz[:, 0] < 0))) + np.rad2deg(np.arctan(xyz[:, 1] / xyz[:, 0]))
    theta_x = (thetax + 90 - 180) % 360
    theta_z = thetaz + 90
    img_cy = np.zeros((180 * res_scale, 360 * res_scale, 1))
    theta_z = np.expand_dims(theta_z,axis=1)
    theta_x = np.expand_dims(theta_x, axis=1)
    return np.concatenate([theta_x*res_scale,theta_z*res_scale],axis=1)
    # img_cy[np.int64(theta_z * res_scale), np.int64(theta_x * res_scale)] = np.expand_dims(255 - depth_val, 1)