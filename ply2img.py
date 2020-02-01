import open3d as od
import copy
import numpy as np
import os
import shutil
import cv2

DATA_FOLDER_PATH = '../Data/Output'
LIDAR_PATH = os.path.join(DATA_FOLDER_PATH,'outlidar')
PROCESSED_PATH = os.path.join(DATA_FOLDER_PATH,'outlidar_proc/')
def get_extrinsic(roll=0,yaw=0,pitch=0,x=0,y=0,z=0):
    # roll = np.radians(transform.rotation.roll)
    # yaw = np.radians(transform.rotation.yaw+90)
    # pitch = np.radians(transform.rotation.pitch+90)
    R_r = np.matrix([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)],
    ])
    R_p = np.matrix([
        [1, 0, 0],
        [0, np.cos(yaw), np.sin(yaw)],
        [0, -np.sin(yaw), np.cos(yaw)],
    ])
    R_y = np.matrix([
        [np.cos(pitch), -np.sin(pitch), 0],
        [np.sin(pitch), np.cos(pitch), 0],
        [0, 0, 1],
    ])

    R = R_y * R_p * R_r
    a = np.concatenate((R,[[x],[y],[z]]),axis=1)
    b = np.concatenate((a,[[0,0,0,1]]),axis=0)
    return b
if __name__ == '__main__':
    if os.path.exists(PROCESSED_PATH):
        shutil.rmtree(PROCESSED_PATH)
    target_folder_path = LIDAR_PATH
    save_folder_path_cy = os.path.join(PROCESSED_PATH,'cy')
    save_folder_path_pinhole = os.path.join(PROCESSED_PATH,'pinhole')
    os.makedirs(save_folder_path_pinhole)
    os.makedirs(save_folder_path_cy)
    agents_folder_list = os.listdir(target_folder_path)
    WINDOW_WIDTH =800
    WINDOW_HEIGHT = 600
    FOV = 150
    res_scale = 2
    for agent_folder in agents_folder_list:
        data_folder_path = os.path.join(target_folder_path,agent_folder)
        ply_file_names = os.listdir(data_folder_path)
        save_folder_path_pinhole_agent = os.path.join(save_folder_path_pinhole,agent_folder)
        save_folder_path_cy_agent = os.path.join(save_folder_path_cy,agent_folder)
        os.makedirs(save_folder_path_pinhole_agent)
        os.makedirs(save_folder_path_cy_agent)
        for file_name in ply_file_names:
            ply_file_path = os.path.join(data_folder_path,file_name)
            save_file_path_cy = os.path.join(save_folder_path_cy_agent,file_name+'.png')

            save_file_path_pinhole = os.path.join(save_folder_path_pinhole_agent, os.path.splitext(file_name)[0]+'.png')
            print(os.path.abspath(ply_file_path))
            pcd_load = od.read_point_cloud(ply_file_path)
            if pcd_load.is_empty():
                continue
            xyz_load = np.asarray(pcd_load.points)
            norm = np.sqrt(np.sum(xyz_load**2,axis=1))
            depth_val = 255 * (norm ) / 120
            depth_val = 32*np.log2(depth_val+1)
            print('%s Norm %d'%(ply_file_path,max(norm[:])))
            if np.isclose(max(norm[:]),0):
                'Error With File'
                continue
            thetaz = np.rad2deg(np.arcsin(xyz_load[:,2]/norm))
            thetax = (180*(np.int64(xyz_load[:,0]<0)))+np.rad2deg(np.arctan(xyz_load[:,1]/xyz_load[:,0]))
            theta_x = thetax+90-180%360
            theta_z = thetaz+90
            img_cy = np.zeros((180 * res_scale, 360 * res_scale, 1))
            img_cy[np.int64(theta_z*res_scale),np.int64(theta_x*res_scale)] = np.expand_dims(255-depth_val,1)
            img_od_cyl = od.Image(img_cy.astype(np.uint8))
            xyz_tmp = np.zeros_like(xyz_load)
            xyz_tmp[:, 1] = xyz_load[:, 0]
            xyz_tmp[:, 2] = xyz_load[:, 1]
            xyz_tmp[:, 0] = xyz_load[:, 2]
            ext = get_extrinsic(roll=0 * np.pi / 4, yaw=4 * np.pi / 4, pitch=2 * np.pi / 4, x=0, y=0, z=0)
            xyz_load = xyz_tmp
            K = np.identity(3)
            K[0, 2] = WINDOW_WIDTH / 2.0
            K[1, 2] = WINDOW_HEIGHT / 2.0
            K[0, 0] = K[1, 1] = 1*WINDOW_WIDTH / (2.0 * np.tan(FOV * np.pi / 360.0))
            xyz = xyz_load
            xyz = np.transpose(xyz)
            xyz = np.concatenate([xyz, np.ones([1, xyz.shape[1]])], axis=0)
            xyz_t = np.matmul(ext,xyz)
            xyz_t = xyz_t[0:3]
            pos2d_scaled =np.matmul(K, xyz_t)
            pos2d = np.array([
                pos2d_scaled[0] / pos2d_scaled[2],
                pos2d_scaled[1] / pos2d_scaled[2],
                pos2d_scaled[2]
            ])
            pos2d = np.squeeze(pos2d)
            flags = (pos2d[2]>0) * (pos2d[0] >= 0) * (pos2d[1] < WINDOW_HEIGHT) * (pos2d[1] >= 0) * (pos2d[0] < WINDOW_WIDTH)
            idx = np.int64(pos2d[:,flags])
            norm = norm[flags]
            img_pinhole = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 1))
            if not norm.size==0:
                depth_val = 255 * (norm-min(norm)) / 120
                depth_val = 255-32 * np.uint8(np.log2(depth_val + 1))

                img_pinhole[idx[1,:],idx[0,:]] = np.expand_dims(depth_val,1)
                # img_pinhole = np.transpose(img_pinhole,[1,0,2])


            print(save_folder_path_pinhole_agent)
            cv2.imwrite(save_file_path_pinhole,img_pinhole)
            # img_od_pinhole = od.Image(img_pinhole.astype(np.uint8))
            od.write_image(save_file_path_cy, img_od_cyl)
            # od.write_image(save_file_path_pinhole, img_od_pinhole)
            # od.draw_geometries([img_od_pinhole])

