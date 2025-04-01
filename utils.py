import torch
import torch.nn as nn
import math
import cv2
import os
import numpy as np
from torch.nn import init
from torch.utils.data.dataset import Dataset

class PositionEmbeddingSine(nn.Module):
    """
    DETR的位置编码。
    num_pos_feats = f_dim // 2,即特征图通道数一般
    在x,y方向分别计算位置编码；奇数位置使用sin编码，偶数位置使用cos编码。以x方向为例，形成num_pos_feats张特征图：
    pos_x = 每一列相同的pos_x, 是第i列： i / w * 2pi
    第k个通道.k为偶数时：pos_x = cos(pos_x / 10000 ^ (k / num_pos_feats))
    k为奇数， pos_x = sin(pos_x / 10000 ^ ((k-1) / num_pos_feats))
    从而不同x位置编码不同，且不同通道位置编码也不同。
    希望这里有两种编码方式，使得累积运动图的特征语义图和空间信息特征语义图能够有不同的位置编码。简单的对调sin和cos即可。
    """
    def __init__(self, num_pos_feats=64, temperature=10000, mode=0, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        '''
        Input:
        x: (b, c, h, w) 图像
        not_mask: position to embed
        '''
        not_mask = (x >= 0)
        not_mask = not_mask.squeeze(dim=1)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)#纵向torch.Size([2, 256, 256])
        x_embed = not_mask.cumsum(2, dtype=torch.float32)#横向
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale#归一化到0-2pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) #num_pos_feats=256 dim_t.shape=256

        pos_x = x_embed[:, :, :, None] / dim_t #torch.Size([2, 256, 256, 256])
        pos_y = y_embed[:, :, :, None] / dim_t #torch.Size([2, 256, 256, 256])
        if self.mode == 0:
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        else: #位置编码也求导。
            # pos_x = torch.stack((pos_x[:, :, :, 0::2].cos(), pos_x[:, :, :, 1::2].sin()), dim=4).flatten(3)
            # pos_y = torch.stack((pos_y[:, :, :, 0::2].cos(), pos_y[:, :, :, 1::2].sin()), dim=4).flatten(3)
            # pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            pos_x = torch.stack((pos_x[:, :, :, 0::2].cos() / dim_t[0::2], -pos_x[:, :, :, 1::2].sin() / dim_t[1::2]),
                                dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].cos() / dim_t[0::2], -pos_y[:, :, :, 1::2].sin() / dim_t[1::2]),
                                dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos #torch.Size([2, 512, 256, 256])


def get_topk_masks(images, n, device, threshold=0.5):
    #从proposal中获取topk个可能的像素点，制作mask
    bs, _, h, w = images.shape
    # 将图像展平为一维数组
    flattened_images = images.view(bs, -1)

    # 找到大于threshold的像素点的位置
    masks = torch.zeros_like((flattened_images), device=device, dtype=bool)
    for i in range(0, bs):
        flattened_image = flattened_images[i]
        # above_threshold = flattened_image > threshold
        # above_threshold_values = flattened_image[above_threshold]
        # above_threshold_indices = torch.nonzero(above_threshold, as_tuple=False)
        #
        # # 如果大于threshold的像素点数量超过n个，选择前n个像素点的索引；否则选择所有的像素点索引
        # if above_threshold_indices.shape[0] > n:
        #     top_n_values, top_n_indices = torch.topk(above_threshold_values, n, largest=True)
        #     top_n_indices = above_threshold_indices[top_n_indices]
        #     mask = torch.zeros_like((flattened_image), device=device, dtype=bool)
        #     mask[top_n_indices[:, 0]] = 1
        #     # mask = mask.view(1, h, w)
        # elif above_threshold_indices.shape[0] > 0:
        #     # print('here2')
        #     top_n_indices = above_threshold_indices
        #     mask = torch.zeros_like((flattened_image), device=device, dtype=bool)
        #     mask[top_n_indices[:, 0]] = 1
        #     # mask = mask.view(1, h, w)
        # else:
            # print('here3')
        top_n_indices = torch.topk((flattened_image), n, dim=0)[1].to(device)
        mask = torch.zeros_like((flattened_image), device=device, dtype=bool)
        mask[top_n_indices] = 1
            # mask = mask.view(1, h, w)
        masks[i] = mask
    # # # 找到每张图像的最大的n个像素点的索引
    # # top_n_indices = torch.topk(flattened_images, n, dim=1)[1].to(device)
    # # 创建一个全零的mask张量
    # mask = torch.zeros_like((flattened_images), device=device, dtype=bool)
    # # # 将最大的n个像素点的位置设为1
    # # mask.scatter_(1, top_n_indices, 1)
    # # 将选定的像素点位置设为1
    # mask[top_n_indices[:, 0], top_n_indices[:, 1]] = 1
    # # 将mask还原成原始图像的形状
    masks = masks.view(bs, 1, h, w)

    for i in range(0, bs):
        # print('sum', torch.sum(masks[i][0]))
        if torch.sum(masks[i][0]) == 0:
            np.savetxt("/home/lab1102/xuyang/SPSA_NET/SPSA_NET/SPSA-OUTPUT/cache/mask_inutils.txt",
                       masks[i][0].cpu().numpy())
            print('torch.isnan(output).any()')
            raise ValueError('nan')

    return masks

def get_region_words(fea_images, masks, max_words_num):
    #fea_images [bs, C, h, w] ; masks [bs, 1, h, w]
    #利用mask和语义特征图，获取视觉单词序列
    bs = fea_images.shape[0]
    f_dim = fea_images.shape[1]
    fea_images = fea_images.permute(0, 2, 3, 1)
    region_words_pad = torch.zeros((bs, max_words_num, f_dim), device=fea_images.device, dtype=fea_images.dtype)
    region_mask = torch.zeros((bs, max_words_num), device=fea_images.device, dtype=torch.bool)
    region_words = []
    for fea_image, mask in zip(fea_images, masks):
        mask_ = mask.expand(f_dim, mask.shape[1], mask.shape[2])
        mask_ = mask_.permute(1, 2, 0)
        mask_float = mask_.float()
        region_word = fea_image * mask_float
        region_word = region_word[mask_[:, :, 0]]
        region_words.append(region_word)
    for i, w in enumerate(region_words):
        l, _ = w.shape
        region_words_pad[i, :l, :] = w
        region_mask[i, :l] = True
    return region_words_pad, region_mask

def get_region_poses(pose_maps, masks, max_words_num):
    #pose_maps [bs, C, h, w] ; masks [bs, 1, h, w]
    bs = pose_maps.shape[0]
    f_dim = pose_maps.shape[1]
    pose_maps = pose_maps.permute(0, 2, 3, 1)
    region_poses_pad = torch.zeros((bs, max_words_num, f_dim), device=pose_maps.device, dtype=pose_maps.dtype)
    region_words = []
    for pose_map, mask in zip(pose_maps, masks):
        mask_ = mask.expand(f_dim, mask.shape[1], mask.shape[2])
        mask_ = mask_.permute(1, 2, 0)
        mask_float = mask_.float()
        region_word = pose_map * mask_float
        region_word = region_word[mask_[:, :, 0]]
        region_words.append(region_word)
    for i, w in enumerate(region_words):
        l, _ = w.shape
        region_poses_pad[i, :l, :] = w

    return region_poses_pad


def getInput(dir, i):
    img = cv2.imread(os.path.join(dir, 'img/' + str('%04d' % i) + '.jpg'), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(dir, 'mask/' + str('%04d' % i) + '.jpg'), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype('float32') / 255
    img = img.astype('float32') / 255
    shape = np.shape(img)

    if i == 0:
        flow = np.zeros([shape[0], shape[1], 2], dtype=np.float32)
        # flowMulti = np.zeros([shape[0], shape[1], 2], dtype=np.float32)
    else:
        flow = np.fromfile(os.path.join(dir, 'flow/' + str('%04d' % i) + '.bin'), dtype=np.float32)
        flow.shape = shape[0], shape[1], 2
        # flowMulti = np.fromfile(os.path.join(dir, 'flowMulti/' + str('%04d' % i) + '.bin'), dtype=np.float32)
        # flowMulti.shape = shape[0], shape[1], 2

    flow_add_path = os.path.join(dir, 'flow_mean_dis/flow_mdis/' + str('%04d' % i) + '.bin')
    if os.path.exists(flow_add_path):
        flow_add = np.fromfile(os.path.join(dir, 'flow_mean_dis/flow_mdis/' + str('%04d' % i) + '.bin'), dtype=np.float32)
        flow_add.shape = shape[0], shape[1], 2
        # print('here')
    else:
        flow_add = flow

    return img, mask, flow, flow_add, flow_add

# 把img经过flow流动后输出，flow类型：np，img类型：np
def flowMove(img, flow, index):
    shape = np.shape(flow)
    if index == 0:
        return np.zeros([shape[0], shape[1]], dtype='float32')
    ret = np.zeros_like(img)
    for i in range(shape[0]):
        for j in range(shape[1]):
            flowij = flow[i, j, :]
            flowi = round(flowij[0])
            flowj = round(flowij[1])
            movei = i - flowj  # 此处应为行i减x方向位移movej，而不是i-movei！
            movej = j - flowi
            if movei < 0 or movei >= shape[0]:
                continue
            if movej < 0 or movej >= shape[1]:
                continue
            ret[i, j] = img[int(movei), int(movej)]
    return ret

class TrainLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainLoader, self).__init__()
        self.dir = dataset_dir
        self.num = len(os.listdir(self.dir))

    def __getitem__(self, index):
        #返回的是第index个序列的文件夹
        return os.path.join(self.dir, os.listdir(self.dir)[index])

    def __len__(self):
        #其实是数据序列的个数
        return self.num


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)

def split_connected_components(bin_img, distance_threshold):
    # Find connected components
    num_labels, labels = cv2.connectedComponents(bin_img)

    # Create a list to store segmented masks
    segmented_masks = []

    # Initialize dictionary to store label coordinates
    label_coords = {}

    # Loop through connected components
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        # Calculate center of mass of the component
        m = cv2.moments(component_mask)
        if m["m00"] != 0:
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
            # Check if the component is far enough from existing components
            is_far = True
            for existing_label, (x, y) in label_coords.items():
                dist = np.sqrt((cX - x) ** 2 + (cY - y) ** 2)
                if dist < distance_threshold:
                    is_far = False
                    break
            if is_far:
                label_coords[label] = (cX, cY)
                segmented_masks.append(component_mask)

    return segmented_masks