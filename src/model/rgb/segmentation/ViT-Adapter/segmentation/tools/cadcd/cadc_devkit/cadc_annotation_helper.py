#!/usr/bin/env python

import numpy as np
import load_calibration
from scipy.spatial.transform import Rotation as R

cam = '0'

def build_objects_from_3d_annotation(cuboids, calib_path):
    objects = []
    
    calib = load_calibration.load_calibration(calib_path)
    # Projection matrix from camera to image frame
    T_IMG_CAM = np.eye(4);
    T_IMG_CAM[0:3,0:3] = np.array(calib['CAM0' + cam]['camera_matrix']['data']).reshape(-1, 3);
    T_IMG_CAM = T_IMG_CAM[0:3,0:4]; # remove last row

    T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + cam]));

    img_h, img_w = 1024, 1280
    
    for cuboid in cuboids:
        if cuboid['points_count'] < 20:
            continue
        
        if cuboid['label'] in ['Car', 'Truck', 'Bus', 'Bicycle', 'Horse and Buggy', 'Garbage Container on Wheels']:
            label = 'vehicle'
        elif cuboid['label'] in ['Pedestrian', 'Pedestrian with Object']:
            label = 'person'
        else:
            continue
        
        T_Lidar_Cuboid = np.eye(4);
        T_Lidar_Cuboid[0:3,0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_matrix();
        T_Lidar_Cuboid[0][3] = cuboid['position']['x'];
        T_Lidar_Cuboid[1][3] = cuboid['position']['y'];
        T_Lidar_Cuboid[2][3] = cuboid['position']['z'];
        
        width = cuboid['dimensions']['x'];
        length = cuboid['dimensions']['y'];
        height = cuboid['dimensions']['z'];

        # Create circle in middle of the cuboid
        tmp = np.matmul(T_CAM_LIDAR, T_Lidar_Cuboid);
        
        if tmp[2][3] < 0: # Behind camera
            continue;
        
        front_right_bottom = np.array([[1,0,0,length/2],[0,1,0,-width/2],[0,0,1,-height/2],[0,0,0,1]]);
        front_right_top = np.array([[1,0,0,length/2],[0,1,0,-width/2],[0,0,1,height/2],[0,0,0,1]]);
        front_left_bottom = np.array([[1,0,0,length/2],[0,1,0,width/2],[0,0,1,-height/2],[0,0,0,1]]);
        front_left_top = np.array([[1,0,0,length/2],[0,1,0,width/2],[0,0,1,height/2],[0,0,0,1]]);

        back_right_bottom = np.array([[1,0,0,-length/2],[0,1,0,-width/2],[0,0,1,-height/2],[0,0,0,1]]);
        back_right_top = np.array([[1,0,0,-length/2],[0,1,0,-width/2],[0,0,1,height/2],[0,0,0,1]]);
        back_left_bottom = np.array([[1,0,0,-length/2],[0,1,0,width/2],[0,0,1,-height/2],[0,0,0,1]]);
        back_left_top = np.array([[1,0,0,-length/2],[0,1,0,width/2],[0,0,1,height/2],[0,0,0,1]]);

        # Project to image
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_right_bottom));
        if tmp[2][3] < 0:
            continue;
        f_r_b = np.matmul(T_IMG_CAM, tmp);
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_right_top));
        if tmp[2][3] < 0:
            continue;
        f_r_t = np.matmul(T_IMG_CAM, tmp);
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_left_bottom));
        if tmp[2][3] < 0:
            continue;
        f_l_b = np.matmul(T_IMG_CAM, tmp);
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_left_top));
        if tmp[2][3] < 0:
            continue;
        f_l_t = np.matmul(T_IMG_CAM, tmp);

        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_right_bottom));
        if tmp[2][3] < 0:
            continue;
        b_r_b = np.matmul(T_IMG_CAM, tmp);
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_right_top));
        if tmp[2][3] < 0:
            continue;
        b_r_t = np.matmul(T_IMG_CAM, tmp);
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_left_bottom));
        if tmp[2][3] < 0:
            continue;
        b_l_b = np.matmul(T_IMG_CAM, tmp);
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_left_top));
        if tmp[2][3] < 0:
            continue;
        b_l_t = np.matmul(T_IMG_CAM, tmp);
        
        # Make sure the 
        # Remove z
        f_r_b_coord = (int(f_r_b[0][3]/f_r_b[2][3]), int(f_r_b[1][3]/f_r_b[2][3]));
        f_r_t_coord = (int(f_r_t[0][3]/f_r_t[2][3]), int(f_r_t[1][3]/f_r_t[2][3]));
        f_l_b_coord = (int(f_l_b[0][3]/f_l_b[2][3]), int(f_l_b[1][3]/f_l_b[2][3]));
        f_l_t_coord = (int(f_l_t[0][3]/f_l_t[2][3]), int(f_l_t[1][3]/f_l_t[2][3]));
        
        if f_r_b_coord[0] < 0 or f_r_b_coord[0] > img_w or f_r_b_coord[1] < 0 or f_r_b_coord[1] > img_h:
            continue;
        if f_r_t_coord[0] < 0 or f_r_t_coord[0] > img_w or f_r_t_coord[1] < 0 or f_r_t_coord[1] > img_h:
            continue;
        if f_l_b_coord[0] < 0 or f_l_b_coord[0] > img_w or f_l_b_coord[1] < 0 or f_l_b_coord[1] > img_h:
            continue;
        if f_l_t_coord[0] < 0 or f_l_t_coord[0] > img_w or f_l_t_coord[1] < 0 or f_l_t_coord[1] > img_h:
            continue;

        b_r_b_coord = (int(b_r_b[0][3]/b_r_b[2][3]), int(b_r_b[1][3]/b_r_b[2][3]));
        b_r_t_coord = (int(b_r_t[0][3]/b_r_t[2][3]), int(b_r_t[1][3]/b_r_t[2][3]));
        b_l_b_coord = (int(b_l_b[0][3]/b_l_b[2][3]), int(b_l_b[1][3]/b_l_b[2][3]));
        b_l_t_coord = (int(b_l_t[0][3]/b_l_t[2][3]), int(b_l_t[1][3]/b_l_t[2][3]));
        
        if b_r_b_coord[0] < 0 or b_r_b_coord[0] > img_w or b_r_b_coord[1] < 0 or b_r_b_coord[1] > img_h:
            continue;
        if b_r_t_coord[0] < 0 or b_r_t_coord[0] > img_w or b_r_t_coord[1] < 0 or b_r_t_coord[1] > img_h:
            continue;
        if b_l_b_coord[0] < 0 or b_l_b_coord[0] > img_w or b_l_b_coord[1] < 0 or b_l_b_coord[1] > img_h:
            continue;
        if b_l_t_coord[0] < 0 or b_l_t_coord[0] > img_w or b_l_t_coord[1] < 0 or b_l_t_coord[1] > img_h:
            continue;

        all_X = [f_r_b_coord[0], f_r_t_coord[0], f_l_b_coord[0], f_l_t_coord[0], b_r_b_coord[0], b_r_t_coord[0], b_l_b_coord[0], b_l_t_coord[0]]
        all_Y = [f_r_b_coord[1], f_r_t_coord[1], f_l_b_coord[1], f_l_t_coord[1], b_r_b_coord[1], b_r_t_coord[1], b_l_b_coord[1], b_l_t_coord[1]]

        min_X = min(all_X)
        min_Y = min(all_Y)
        max_X = max(all_X)
        max_Y = max(all_Y)
        
        coord_1 = [min_X, min_Y]
        coord_2 = [min_X, max_Y]
        coord_3 = [max_X, max_Y]
        coord_4 = [max_X, min_Y]
        
        polygon = [coord_1, coord_2, coord_3, coord_4]
        
        object = {}
        object['label'] = label
        object['polygon'] = polygon
        
        objects.append(object)
        
    return objects