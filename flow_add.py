# RMPE module
# use this get RMM. now we need to do this before train GST-Net
import os
import numpy as np
import cv2
import heapq

flow_move_matrix = np.zeros([256, 256, 2], dtype=np.float32)
def correct_flow(flow, shape, flow_last, beta):
    global flow_move_matrix
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    flow_move_matrix = flow_move_matrix + flow
    move_i = y - np.round(flow_move_matrix[:, :, 1])
    move_j = x - np.round(flow_move_matrix[:, :, 0])
    flow_move_matrix = np.where(np.abs(flow_move_matrix) > 0.5, 0, flow_move_matrix)

    valid_coordinates = (move_i >= 0) & (move_i < shape[0]) & (move_j >= 0) & (move_j < shape[1])

    move_last = flow_last[move_i[valid_coordinates].astype(int), move_j[valid_coordinates].astype(int), :]

    flow_new = flow.copy()
    flow_new[valid_coordinates] += beta * move_last

    return flow_new

def sort_by_number(file_name):
    return int(file_name[:-4])

# 获取每个通道上的topk个像素点的坐标
def get_topk_coordinates(data, topk):
    channel1_feature_map = data[:, :, 0]
    channel2_feature_map = data[:, :, 1]

    channel1_topk_coordinates = np.argpartition(channel1_feature_map.flatten(), -topk)[-topk:]
    channel2_topk_coordinates = np.argpartition(channel2_feature_map.flatten(), -topk)[-topk:]

    channel1_topk_coordinates = np.column_stack(np.unravel_index(channel1_topk_coordinates, channel1_feature_map.shape))
    channel2_topk_coordinates = np.column_stack(np.unravel_index(channel2_topk_coordinates, channel2_feature_map.shape))

    return channel1_topk_coordinates, channel2_topk_coordinates

def add_flow(flow_path, out_path):
    global flow_move_matrix
    flows_names = os.listdir(flow_path)
    shape = (256, 256)
    top_k = 1000
    # flow_to_add = []
    # sign_to_add = []
    sorted_flows_names = sorted(flows_names, key=lambda x: sort_by_number(x))
    flow_add = np.zeros([shape[0], shape[1], 2], dtype=np.float32)
    direction_add = np.zeros([shape[0], shape[1], 2], dtype=np.float32)
    for i, flow_name in enumerate(sorted_flows_names):
        # flow_add = np.zeros([shape[0], shape[1], 2], dtype=np.float32)
        # sign_add = np.zeros([shape[0], shape[1], 2], dtype=np.float32)
        if i == 0:
            flow = np.zeros([shape[0], shape[1], 2], dtype=np.float32)
            flow.shape = shape[0], shape[1], 2
        else:
            # print(os.path.join(flow_path, flow_name))
            flow = np.fromfile(os.path.join(flow_path, flow_name), dtype=np.float32)
            flow.shape = shape[0], shape[1], 2

        # ... get flow_add, (RMM), It is not currently available.
        # ...
        # ...

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros([shape[0], shape[1], 3], dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if not os.path.exists(out_path + '/vision/'):
            os.makedirs((out_path + '/vision/'))
        cv2.imwrite(out_path + '/vision/' + str(i).zfill(4) + '.jpg', bgr)

        mag, ang = cv2.cartToPolar(flow_add[..., 0], flow_add[..., 1])
        hsv = np.zeros([shape[0], shape[1], 3], dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if not os.path.exists(out_path + '/vision1/'):
            os.makedirs((out_path + '/vision1/'))
        cv2.imwrite(out_path + '/vision1/' + str(i).zfill(4) + '.jpg', bgr)

        if not os.path.exists(out_path + '/flow_mdis/'):
            os.makedirs(out_path + '/flow_mdis/')
        flow_add.tofile(out_path + '/flow_mdis/' + '' + str(i).zfill(4) + '.bin')

if __name__ == '__main__':
    datasets_dir = 'IDSAT'  # path to datasets_dirs, in which are img sequence dirs
    dir_names = os.listdir(datasets_dir)
    for index, dir_name in enumerate(dir_names):
        print(dir_name)
        dir_path = os.path.join(datasets_dir, dir_name)
        flow_path = os.path.join(dir_path, 'flow')
        out_path = os.path.join(dir_path, 'flow_mean_dis')
        add_flow(flow_path, out_path)








