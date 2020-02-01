import cv2
import os
import numpy as np
DATA_FOLDER_PATH = '../Data/Output'
LIDAR_PATH = os.path.join(DATA_FOLDER_PATH,'outlidar')
PROCESSED_PATH = os.path.join(DATA_FOLDER_PATH,'outlidar_proc/')
image_folder = 'images'


# if os.path.exists(PROCESSED_PATH):
#     shutil.rmtree(PROCESSED_PATH)
target_folder_path = LIDAR_PATH
save_folder_path_cy = os.path.join(PROCESSED_PATH, 'cy')
save_folder_path_pinhole = os.path.join(PROCESSED_PATH, 'pinhole')
# os.makedirs(save_folder_path_pinhole)
# os.makedirs(save_folder_path_cy)
agents_folder_list = os.listdir(target_folder_path)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FOV = 170
res_scale = 2
for agent_folder in agents_folder_list:
    data_folder_path = os.path.join(target_folder_path, agent_folder)

    save_folder_path_pinhole_agent = os.path.join(save_folder_path_pinhole, agent_folder)
    save_folder_path_cy_agent = os.path.join(save_folder_path_cy, agent_folder)
    files_cy = os.listdir(save_folder_path_cy_agent)
    files_ph = os.listdir(save_folder_path_pinhole_agent)
    # os.makedirs(save_folder_path_pinhole_agent)
    # os.makedirs(save_folder_path_cy_agent)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    print(save_folder_path_cy_agent)
    video_name_cy = '../Data/video_%s_cy.wmf'%agent_folder
    video_name_pin = '../Data/video_%s_pin.wmf' % agent_folder
    images_cy = [img for img in files_cy if img.endswith(".png")]
    images_ph = [img for img in files_ph if img.endswith(".png")]
    frame = cv2.imread(os.path.join(save_folder_path_cy_agent, images_cy[0]))
    height, width, layers = frame.shape
    video_cy = cv2.VideoWriter(video_name_cy, 0, 15, (width, height))
    frame = cv2.imread(os.path.join(save_folder_path_pinhole_agent, images_ph[0]))
    height, width, layers = frame.shape

    for idx,file_name in enumerate(images_cy):
        print(file_name)
        if not file_name.endswith(".png"):
            continue
        # print(file_name)
        frame = cv2.imread(os.path.join(save_folder_path_cy_agent,file_name))
        frame = np.uint8((frame - np.min(frame)/np.max(frame))*255)
        video_cy.write(frame)
        if idx==1000:
            break
    cv2.destroyAllWindows()
    video_cy.release()
    video_ph = cv2.VideoWriter(video_name_pin, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (width, height))

    for idx,file_name in enumerate(images_ph):
        print(file_name)
        if not file_name.endswith(".png"):
            continue
        # print(file_name)
        frame = cv2.imread(os.path.join(save_folder_path_pinhole_agent,file_name))
        frame = np.uint8((frame - np.min(frame) / np.max(frame)) * 255)
        video_ph.write(frame)
        if idx==1000:
            break
    cv2.destroyAllWindows()
    video_cy.release()
