import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from loss import SoftIoULoss
from torch.utils.data import DataLoader
import skimage.io as io

from SPSA_Net import SPSA_Net

from evaluateFunc import SegmentationMetric
from utils import *

import warnings

warnings.filterwarnings('ignore')
predict_times = 500
times = []
now_time = 0

def test(args):
    val_datasets = TrainLoader(dataset_dir=args.testPath)
    val_loader = DataLoader(val_datasets, batch_size=1, shuffle=False)
    val_step_num = len(val_loader)
    print("using {} batch fot validation.".format(val_step_num))

    model = SPSA_Net(topk=args.TOPK, f_dim=args.f_dim, in_channels=args.in_channels, GCAN_chs=args.GCAN_chs,
                     trans_encode_layers=args.multi_head_num)

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    device_ids = [0]
    device = device_ids
    print("the validating process from %s" % args.resume)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    validate(args, args.img_out_path, model, val_loader, device, val_step_num, iou_tho=0.33)

    print('Finished Validating!')


def validate(args, img_out_path, model, val_loader, device, val_step_num, iou_tho=0.33):
    global predict_times, times, now_time
    # validate
    model.eval()
    val_loss = 0.0
    TP_num = 0
    D_num = 0
    target_num = 0
    total_img_num = 0

    with torch.no_grad():
        total_metric = SegmentationMetric(2)
        total_metric.reset()
        for step, dirs in enumerate(val_loader, start=0):
            batch_target_num = 0
            batch_D_num = 0
            batch_TP_num = 0
            batch_img_num = float('inf')
            pred = np.zeros([len(dirs), 256, 256], dtype=np.float32)
            probability_maps = np.zeros([len(dirs), 256, 256], dtype=np.float32)
            for d in range(len(dirs)):
                batch_img_num = min(batch_img_num, len(os.listdir(os.path.join(dirs[d], 'img'))))
            total_img_num += batch_img_num

            for i in range(batch_img_num):
                print("\r\tValidate step:%04d/%04d,processing %d"
                      % (step, val_step_num, i), end="", flush=True)
                # 两种堆叠方法，1：列表堆叠后stack;2：对空tensor进行cat:flows = torch.Tensor(),flows.cat((flows, img),dim=0)
                imagesList = []
                accumulation_List = []
                labelsList = []
                for j in range(len(dirs)):
                    img, mask, flow, flow_add, flow_mutil = getInput(dirs[j], i)
                    imagesList.append(torch.from_numpy(np.array(img)))
                    mask = torch.from_numpy(mask)
                    labelsList.append(torch.unsqueeze(mask, 0))
                    accumulation_List.append(torch.from_numpy(flow_add).permute(2, 0, 1))

                images = torch.stack(imagesList, dim=0)  # 形成batch [batch,channel,height,width]
                accumulations = torch.stack(accumulation_List, dim=0)
                labels = torch.stack(labelsList, dim=0)

                images = images.unsqueeze(1)

                images = images.cuda()
                accumulations = accumulations.cuda()
                labels = labels.cuda()

                # start_time = time.time()
                stout = model(accumulations, images)

                # end_time = time.time()
                # infertime = start_time - end_time
                # times.append(infertime)
                # now_time = now_time + 1
                # if now_time == predict_times:
                #     times = sorted(times)
                #     times = times[20:predict_times - 20]
                #     mean_time = sum(times) / len(times)
                #     print('mean_time', mean_time)

                loss = SoftIoULoss(stout, labels)
                val_loss += loss.item()

                output = stout[0, 0, :, :].detach().cpu().numpy()
                out_img = output * 255

                pathName = os.path.join(img_out_path, os.path.basename(dirs[0]), "org")
                if not os.path.exists(pathName):
                    os.makedirs(pathName)
                io.imsave(pathName + '/' + str('%04d' % i) + ".jpg", out_img.astype(np.uint8))

                max_threshold = max(0.7 * np.max(output), 0.5 * np.std(output) + np.mean(output))
                _, out_put = cv2.threshold(output, max_threshold, 1, cv2.THRESH_BINARY)
                out_img = out_put * 255  # 去除维数为1的channal
                pathName = os.path.join(img_out_path, os.path.basename(dirs[0]), "bw")
                if not os.path.exists(pathName):
                    os.makedirs(pathName)
                io.imsave(pathName + '/' + str('%04d' % i) + ".jpg", out_img.astype(np.uint8))

                # 计算指标
                out_put = out_put.astype(np.uint8)
                labels = np.rint(labels.squeeze().cpu().numpy()).astype(np.uint8)

                # 像素级指标
                total_metric.addBatch(out_put, labels)

                # 数量级指标
                mask_lists = split_connected_components(labels, 5)
                out_lists = split_connected_components(out_put, 5)
                num_metric = SegmentationMetric(2)
                for ii in range(len(mask_lists)):
                    batch_target_num += 1
                    max_iou = 0
                    max_pa = 0
                    for jj in range(len(out_lists)):
                        batch_D_num += 1
                        num_metric.reset()
                        num_metric.addBatch(out_lists[jj], mask_lists[ii])
                        max_iou = max(max_iou, num_metric.meanIntersectionOverUnion()[1])
                        max_pa = max(max_pa, num_metric.classPixelAccuracy()[1])
                    if max_pa > iou_tho:
                        batch_TP_num += 1
            num_recall = 0 if batch_target_num == 0 else batch_TP_num / batch_target_num
            num_accuracy = 0 if batch_D_num == 0 else batch_TP_num / batch_D_num
            print("\r%s: num_recall:%.4f, num_accuracy:%.4f" %
                  (os.path.split(dirs[0])[1], num_recall, num_accuracy))
            TP_num += batch_TP_num
            D_num += batch_D_num
            target_num += batch_target_num
        val_loss = val_loss / total_img_num
        mIOU = total_metric.meanIntersectionOverUnion()[1]
        pixel_recall = total_metric.classRecall()[1]
        pixel_accuracy = total_metric.classPixelAccuracy()[1]
        num_recall = 0 if target_num == 0 else TP_num / target_num
        num_accuracy = 0 if D_num == 0 else TP_num / D_num
        F1 = 2 * num_recall * num_accuracy / (num_recall + num_accuracy)
        print('\rF1: %.4f mIOU:%.4f pixel_recall:%.4f pixel_accuracy:%.4f num_recall:%.4f num_accuracy:%.4f' %
              (F1, mIOU, pixel_recall, pixel_accuracy, num_recall, num_accuracy))
        return 1-F1


def evaluate_res_(label_path, results_path):
    label_fold_names = [x for x in os.listdir(label_path)]
    total_metric_sum = SegmentationMetric(2)
    total_metric_sum.reset()  # 将混淆矩阵清零
    TP_num = 0
    D_num = 0
    target_num = 0
    for i in range(len(label_fold_names)):
        label_fold_path = os.path.join(label_path, label_fold_names[i], 'mask')

        results_fold_names = [x for x in os.listdir(results_path)]
        for j in range(0, 1):
            result_fold_path = os.path.join(results_path, label_fold_names[i], 'tho')

            print("processing %s %s \n"
                  % (label_fold_names[i], results_fold_names[j]), end="", flush=True)
            _, _, _, _, _, _, TP_num, D_num, target_num = evaluate_res(label_fold_path, result_fold_path,
                                                                       total_metric_sum, TP_num, D_num, target_num)
            print('TP_num:%.4f D_num:%.4f target_num:%.4f' % (TP_num, D_num, target_num))

    mIOU = total_metric_sum.meanIntersectionOverUnion()[1]
    pixel_recall = total_metric_sum.classRecall()[1]
    pixel_accuracy = total_metric_sum.classPixelAccuracy()[1]
    pixel_fa = 1 - total_metric_sum.classRecall()[0]
    num_recall = 0 if target_num == 0 else TP_num / target_num
    num_accuracy = 0 if D_num == 0 else TP_num / D_num
    print(
        '\r all_test_data: mIOU:%.4f pixel_recall:%.4f pixel_accuracy:%.4f num_recall:%.4f num_accuracy:%.4f pixel_fa:%.4f' %
        (mIOU, pixel_recall, pixel_accuracy, num_recall, num_accuracy, pixel_fa))


# 评估二值图像的指标结果，单个文件夹
def evaluate_res(label_path, results_path, total_metric_sum, TP_num, D_num, target_num):
    label_filenames = [x for x in os.listdir(results_path)]
    results_filenames = [x for x in os.listdir(results_path)]

    total_metric = SegmentationMetric(2)
    total_metric.reset()
    batch_target_num = 0
    batch_D_num = 0
    batch_TP_num = 0
    iou_tho = 0.3


    for im_name in range(len(label_filenames)):
        # if im_name>10:
        #     continue
        print("\r\tprocessing %d/%d"
              % (im_name, len(label_filenames)), end="", flush=True)
        label = cv2.imread(os.path.join(label_path, label_filenames[im_name]), cv2.IMREAD_GRAYSCALE)
        result = cv2.imread(os.path.join(results_path, results_filenames[im_name]), cv2.IMREAD_GRAYSCALE)

        label = label.astype('float32') / 255
        result = result.astype('float32') / 255
        labels = np.rint(label).astype(np.uint8)
        output = np.rint(result).astype(np.uint8)

        # 像素级指标
        total_metric.addBatch(output, labels)
        total_metric_sum.addBatch(output, labels)

        # 数量级指标
        mask_lists = split_connected_components(labels, 5)
        out_lists = split_connected_components(output, 5)
        num_metric = SegmentationMetric(2)
        for ii in range(len(mask_lists)):
            batch_target_num += 1
            max_iou = 0
            max_pa = 0
            for jj in range(len(out_lists)):
                batch_D_num += 1
                num_metric.reset()  # 将混淆矩阵清零
                num_metric.addBatch(out_lists[jj], mask_lists[ii])
                max_iou = max(max_iou, num_metric.meanIntersectionOverUnion()[1])
                max_pa = max(max_pa, num_metric.classPixelAccuracy()[1])
            if max_pa > iou_tho:
                batch_TP_num += 1

    TP_num += batch_TP_num
    D_num += batch_D_num
    target_num += batch_target_num
    mIOU = total_metric.meanIntersectionOverUnion()[1]
    pixel_recall = total_metric.classRecall()[1]
    pixel_accuracy = total_metric.classPixelAccuracy()[1]
    pixel_fa = 1 - total_metric.classRecall()[0]
    num_recall = 0 if target_num == 0 else batch_TP_num / batch_target_num
    num_accuracy = 0 if D_num == 0 else batch_TP_num / batch_D_num
    print('\r mIOU:%.4f pixel_recall:%.4f pixel_accuracy:%.4f num_recall:%.4f num_accuracy:%.4f pixel_fa:%.4f' %
          (mIOU, pixel_recall, pixel_accuracy, num_recall, num_accuracy, pixel_fa))
    return mIOU, pixel_recall, pixel_accuracy, num_recall, num_accuracy, pixel_fa, TP_num, D_num, target_num


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
    parser.add_argument("--img_out_path", default='/home/lab1102/xuyang/data/data_xuyang/SPSA_OUTPUT/img_out_path',
                        help="out_put img path")

    args = parser.parse_args()
    # ------------------------------------------------------- #
    args.trainPath = r"datasets\train"
    args.testPath = r'datasets\test\ALL'

    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

    args.output_dir = "SPSA-OUTPUT\local_result" + '/' + formatted_time
    args.resume = ''

    args.TOPK = 1000
    args.multi_head_num = 4
    args.in_channels = 1
    args.GCAN_chs = 16
    args.f_dim = 256
    args.seg_posw = 3

    args.epochs = 40
    args.batch_size = 4
    args.device = "1,2,3,4"
    args.start_epoch = 1
    args.lr = 0.001

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test(args)