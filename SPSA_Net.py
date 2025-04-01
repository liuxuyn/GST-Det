#Global Spatial -Temporal feature fusion network
import torch
import torch.nn as nn
from utils import *
from transformer import Transformer
import torch.nn.functional as F
import cv2

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class fea_add_module(nn.Module):
    def __init__(self, channels):
        super(fea_add_module, self).__init__()

        self.ca1 = ChannelAttention(channels * 2)
        self.ca2 = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels * 2))

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels))

        self.center_layer = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, S, T):
        ST = torch.cat((S, T), dim=1)
        out1 = self.ca1(ST) * self.sa(ST) * ST
        res1 = self.shortcut1(ST)
        out1 += res1
        out2 = self.center_layer(out1)
        res2 = self.shortcut2(out2)
        out = self.ca2(out2) * self.sa(out2) * out2
        out += res2
        out = self.relu(out)
        return out

class out_module(nn.Module):
    def __init__(self, channels=32):
        super(out_module, self).__init__()
        # decoder, which is not currently available.

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, st_seg, so, to, s3, t3, sc, tc, s1, t1, s0, t0):
        # decoder, which is not currently available.
        return sto

class fea_get_module(nn.Module):
    def __init__(self, channels=32, in_channel=1):
        super(fea_get_module, self).__init__()
        # feature encoder, which is not currently available.


    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, s, t):
        # feature encoder, which is not currently available.
        return

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        Norm2d = nn.BatchNorm1d
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.conv_1d_layer = nn.Sequential(
            nn.Conv1d(input_dim * 2, input_dim, kernel_size=3, padding=1),
            nn.Conv1d(input_dim, input_dim, kernel_size=1),
            Norm2d(input_dim)
        )


    def forward(self, x):
        b, l, c = x.shape
        x_s = x[:, :l//2, :]
        x_t = x[:, l//2:, :]
        x_st = torch.cat((x_s, x_t), dim=-1)
        x_st = x_st.permute(0, 2, 1)
        x_st = self.conv_1d_layer(x_st)
        x_st = x_st.permute(0, 2, 1)
        for i, layer in enumerate(self.layers):
            x_st = F.relu(layer(x_st)) if i < self.num_layers - 1 else layer(x_st)
        return x_st

class get_vision_words:
    def __init__(self, proposal_maps, feature_maps, topk_words=200, pose_mode=0):
        for i in [0, 2, 3]:
            if proposal_maps.shape[i] != feature_maps.shape[i]:
                raise ValueError('proposal_map is not match with feature_map')
        self.bs = proposal_maps.shape[0]
        self.feature_chs = feature_maps.shape[1]
        self.h = proposal_maps.shape[2]
        self.w = proposal_maps.shape[3]
        self.topk = topk_words
        self.pose_mode = pose_mode

        self.pro_maps = proposal_maps
        self.f_maps = feature_maps

        self.posembedding = PositionEmbeddingSine(num_pos_feats=self.feature_chs // 2, normalize=True, mode=self.pose_mode)

    def get_region_masks(self):
        self.mask_maps = get_topk_masks(self.pro_maps, self.topk, device=self.f_maps.device)  # [bs, 1, h, w]
        return self.mask_maps

    def get_region_words(self):
        self.region_words, self.region_mask = get_region_words(self.f_maps, self.mask_maps, self.topk)
        return self.region_words, self.region_mask

    def get_embedding_poses(self):
        image_poses = self.posembedding(self.pro_maps)
        self.region_poses = get_region_poses(image_poses, self.mask_maps, self.topk)
        return self.region_poses

class SPSA_Net(nn.Module):
    def __init__(self, topk, f_dim, in_channels, GCAN_chs, num_heads):
        super(SPSA_Net, self).__init__()
        self.topk = topk
        self.f_dim = GCAN_chs * 16
        self.in_channels = in_channels
        self.GCAN_chs = GCAN_chs

        self.num_heads = num_heads
        self.fea_get = fea_get_module(channels=self.GCAN_chs, in_channel=self.in_channels)
        self.transformer = Transformer(nhead=self.num_heads, d_model=self.f_dim)
        self.pro_embed = MLP(self.f_dim, self.f_dim*2, self.f_dim, 3)
        self.output_layers = out_module(channels=self.GCAN_chs)

    def forward(self, T_map, S_map):
        bs, _, h, w = T_map.shape
        bs1, _, h1, w1 = S_map.shape

        if bs != bs1 or h != h1 or w !=w1:
            raise ValueError('two inputs have different shape ')

        S_fea_maps, T_fea_maps = self.fea_get(S_map, T_map)

        # GSTDEM, which is not currently available.
        ST_seg_output_ = self.pro_embed(now_frame_output)
        ST_seg_img = ST_seg_output_.view(bs, self.f_dim, h // 8, w // 8)
        stout = self.output_layers(ST_seg_img, so, to, s3, t3, sc, tc, s1, t1, s0, t0)

        return stout



if __name__ == '__main__':
    # for test only
    bs = 2
    n = 200
    h = 256
    w = 256
    images_S = torch.randint(0, 255, (bs, 1, h, w)).float()
    images_T = torch.randint(0, 255, (bs, 1, h, w)).float()
    images_S = images_S / 255.0
    images_T = images_T / 255.0

    images_S = images_S.cuda()
    images_T = images_T.cuda()
    model = SPSA_Net(topk=200, f_dim=512, in_channels=1, GCAN_chs=32, trans_encode_layers=4)
    model = nn.DataParallel(model).cuda()
    model.train()
    model(images_T, images_S)



























