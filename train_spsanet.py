import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import random
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from loss import SoftIoULoss, MF1_criterion
from SPSA_Net import SPSA_Net
import torch.nn as nn
from test import validate
from utils import *
import warnings

warnings.filterwarnings('ignore')

def train(args):
    print(torch.cuda.device_count())

    train_datasets = TrainLoader(dataset_dir=args.trainPath)
    val_datasets = TrainLoader(dataset_dir=args.testPath)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_datasets, batch_size=nw, shuffle=True, num_workers=8)
    train_step_num = len(train_loader)
    val_loader = DataLoader(val_datasets, batch_size=1, shuffle=False, num_workers=8)
    val_step_num = len(val_loader)
    print("using {} step for training, {} step fot validation.".format(train_step_num,
                                                                         val_step_num))

    model = SPSA_Net(topk=args.TOPK, f_dim=args.f_dim, in_channels=args.in_channels, GCAN_chs=args.GCAN_chs,
                     trans_encode_layers=args.multi_head_num)

    device_ids = [0]
    device = device_ids

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        print("the training process from epoch{}...".format(args.start_epoch))
        epoch_start = args.start_epoch
    else:
        model.apply(weights_init_xavier)
        epoch_start = 1

    model = nn.DataParallel(model).cuda()

    # optimizer
    param_dicts = [
        {"params": [v for k, v in model.named_parameters() if "GCAN" in k and v.requires_grad],
         "lr": args.lr},
        {"params": [v for k, v in model.named_parameters() if "GCAN" not in k and v.requires_grad],
         "lr": args.lr}
    ]
    optimizer = torch.optim.SGD(param_dicts, lr=args.lr)
    
    # optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[16, 32, 48], gamma=0.5)
    best_loss = float('inf')
    for epoch in range(epoch_start, args.epochs):

        # validate
        img_out_path = os.path.join(args.output_dir, 'img_output')
        if (epoch > 0 and epoch % 2 == 0) or epoch == 40:
            val_loss = validate(args, img_out_path, model, val_loader, device, val_step_num, iou_tho=0.33)
            if val_loss < best_loss:  # 必须经过若干轮训练才能进行保存，而不是直接报错
                best_loss = val_loss
                print("in epoch %d get best!" % (epoch - 1))

                # _use_new_zipfile_serialization = False
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(),
                               os.path.join(args.output_dir, 'best_%03d_SPSANet.pth' % (epoch - 1)))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(args.output_dir, 'best_%03d_SPSANet.pth' % (epoch - 1)))
        if epoch % 2 == 0:
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(),
                           os.path.join(args.output_dir, '%03d_%03d_SPSANet_lrdown.pth' % ((epoch-1), args.epochs)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, '%03d_%03d_SPSANet_lrdown.pth' % ((epoch-1), args.epochs)))  # 每个epoch之后保存一次temp

        # train
        model.train()
        scheduler.step()
        print("lr1={}".format(optimizer.state_dict()['param_groups'][0]['lr']),
              "lr2={}".format(optimizer.state_dict()['param_groups'][1]['lr']))
        epoch_loss = 0
        total_img_num = 0
        for step, dirs in enumerate(train_loader, start=0):
            # 初始化
            batch_img_num = float('inf')

            img_num_list = []
            for d in range(len(dirs)):
                batch_img_num = min(batch_img_num, len(os.listdir(os.path.join(dirs[d], 'img'))))
                img_num_list.append(len(os.listdir(os.path.join(dirs[d], 'img'))))
            total_img_num += batch_img_num

            frame_to_start = []
            for d in range(len(dirs)):
                frame_to_start_min = img_num_list[d] - batch_img_num
                frame_2_start = random.randint(0,frame_to_start_min)
                frame_to_start.append(frame_2_start)

            for i in range(batch_img_num):
                imagesList = []
                accumulation_List = []
                labelsList = []
                for j in range(len(dirs)):
                    img, mask, flow, flow_add, flow_mutil = getInput(dirs[j], i + frame_to_start[j])
                    imagesList.append(torch.from_numpy(np.array(img)))
                    mask = torch.from_numpy(mask)
                    labelsList.append(torch.unsqueeze(mask, 0))
                    accumulation_List.append(torch.from_numpy(flow_add).permute(2, 0, 1))


                images = torch.stack(imagesList, dim=0) #形成batch [batch,channel,height,width]
                accumulations = torch.stack(accumulation_List, dim=0)
                labels = torch.stack(labelsList, dim=0)

                images = images.unsqueeze(1)

                if not os.path.exists(args.output_dir + '/TRAIN_IMG_VISION/'):
                    os.makedirs(args.output_dir + '/TRAIN_IMG_VISION/')

                # 训练过程
                images = images.cuda()
                accumulations = accumulations.cuda()
                labels = labels.cuda()

                stout = model(accumulations, images)

                MD, FA, MF_loss_ST = MF1_criterion(stout, labels)
                MF_loss_ST = MD + FA
                gcan_loss = MF_loss_ST * 1000
                iou_loss = SoftIoULoss(stout, labels)
                loss = iou_loss + gcan_loss

                cv2.imwrite(args.output_dir + '/TRAIN_IMG_VISION/' + '%04d_pred.jpg' % i,
                           stout[0, 0, :, :].detach().cpu().numpy() * 255)
                # cv2.imwrite(args.output_dir + '/TRAIN_IMG_VISION/' + '%04d_SGCAN.jpg' % i,
                #            S_pro_maps[0, 0, :, :].detach().cpu().numpy() * 255)
                # cv2.imwrite(args.output_dir + '/TRAIN_IMG_VISION/' + '%04d_TGCAN.jpg' % i,
                #             T_pro_maps[0, 0, :, :].detach().cpu().numpy() * 255)
                # cv2.imwrite(args.output_dir + '/TRAIN_IMG_VISION/' + '%04d_STGCAN.jpg' % i,
                #             ST_pro_maps[0, 0, :, :].detach().cpu().numpy() * 255)
                # cv2.imwrite(args.output_dir + '/TRAIN_IMG_VISION/' + '%04d_predS.jpg' % i,
                #             probability_maps_S[0, :, :] * 255)
                # cv2.imwrite(args.output_dir + '/TRAIN_IMG_VISION/' + '%04d_predT.jpg' % i,
                #             probability_maps_T[0, :, :] * 255)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                rate = (i + 1) / batch_img_num
                a = "*" * int(rate * 50)
                b = "." * (50 - int(rate * 50))
                print("\rEpoch:%04d/%04d step:%04d/%04d," % (epoch, args.epochs, step, train_step_num),
                      "{:^3.0f}%[{}->{}] train loss: {:.4f} iou loss: {:.4f} GCAN loss: {:.4f}".format(int(rate * 100),
                      a, b, loss, iou_loss, gcan_loss), end="", flush=True)
        print("\nTrain loss result: total loss: %.4f " % (epoch_loss / total_img_num))
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'temp_%03d_SPSANet_lrdown.pth' % args.epochs))
        else:
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'temp_%03d_SPSANet_lrdown.pth' % args.epochs))#每个epoch之后保存一次temp

        
    print('Finished Training!')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    # 训练设备类型
    parser.add_argument('--device', default='cuda:2', help='device')
    # 训练数据集目录
    parser.add_argument('--trainPath', default='./', help='datasets')
    # 测试数据集目录
    parser.add_argument('--testPath', default='./', help='datasets')
    # 权重文件保存地址
    parser.add_argument('--output_dir', default='./pth', help='path dir to save')
    # 若要接着训练，则指定上次训练文件保存地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 制定接着从哪论开始训练
    parser.add_argument('--start_epoch', default=1, type=int, help='start epoch')
    # 训练总epoch
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='epoch')
    # 训练的batch size
    parser.add_argument('--batch_size', default=8, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')

    parser.add_argument('--TOPK', type=str, default=200, help='topk visual words')
    parser.add_argument('--multi_head_num', type=int, default=4, help='multi_head_attention number')
    parser.add_argument('--in_channels', type=int, default=1, help='input img channels num')
    parser.add_argument('--GCAN_chs', type=int, default=32, help='base channels of GCAN')
    parser.add_argument('--f_dim', type=int, default=512, help='hidden dims or channels of feature maps')

    parser.add_argument("--seg_posw", type=int, default=3, help="Positive weights of BCEwithLogitLoss for segmentation")

    args = parser.parse_args()
    args.trainPath = r"datasets/new_train"
    args.testPath = r'datasets/new_test/ALL'

    from datetime import datetime
    # 获取当前时间
    current_time = datetime.now()
    # 将时间格式化为字符串
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S") #' '2023-12-31-16-10-09'

    args.output_dir = "SPSA_OUTPUT" + '/' + formatted_time
    args.resume = ''
    args.TOPK = 1024
    args.multi_head_num = 4
    args.in_channels = 1
    args.GCAN_chs = 16
    args.f_dim = args.GCAN_chs * 16
    args.seg_posw = 3

    args.epochs = 64
    args.batch_size = 8
    args.device = "1,2,3,4"
    args.start_epoch = 1
    args.lr = 0.001

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(args)