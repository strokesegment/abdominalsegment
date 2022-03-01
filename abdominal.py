import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import lr_scheduler
from skimage.transform import resize
from torch import optim
# import xlwt
# import xlrd  # 导入模块
import torch.utils.data as Data
# from torch.utils.data import Dataset
# from scipy.misc import toimage
# from PIL import Image
from scipy.io import savemat
import pandas as pd
import torch.cuda

class SubBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.xpn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.xpn_block(x)

class conv2d_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(0.25),
        )
    def forward(self, x):
        return self.conv2d_layer(x)


# Mask GAP
def MaskGAP(sup, sup_mask):
    # support image feature and mask (repeat)
    sup_proto = sup * sup_mask
    # feat_h, feat_w
    feat_h, feat_w = sup_proto.shape[-2:][0], sup_proto.shape[-2:][1]
    # mask_gap
    area = F.avg_pool2d(sup_mask, (sup_proto.size()[2], sup_proto.size()[3])) * feat_h * feat_w + 0.0005
    sup_proto = F.avg_pool2d(input=sup_proto,
                                   kernel_size=sup_proto.shape[-2:]) * feat_h * feat_w / area
    return sup_proto


def cs(sup, que):

    cosine_similarity = F.cosine_similarity(sup, que, dim=1)

    cosine_similarity[cosine_similarity >= 0.5] = 1
    cosine_similarity[cosine_similarity < 0.5] = 0

    return cosine_similarity


def gen_mask(cosine_sim, sup_mask_16):

    new_mask = torch.zeros_like(sup_mask_16)
    for a in range(new_mask.shape[0]):
        for b in range(new_mask.shape[1]):
            for c in range(new_mask.shape[2]):
                if cosine_sim[a, b, c] == 1:
                    if sup_mask_16[a, b, c] == 1:
                        new_mask[a, b, c] = 1
    return new_mask


def qucha(que_mask_16, que_guidance):

    mask_cha = torch.zeros_like(que_mask_16)
    for a in range(mask_cha.shape[0]):
        for b in range(mask_cha.shape[1]):
            for c in range(mask_cha.shape[2]):
                for d in range(mask_cha.shape[3]):
                    if que_mask_16[a, b, c, d] == 1:
                        if que_guidance[a, b, c, d] == 0:
                            mask_cha[a, b, c, d] = 1
                    if que_mask_16[a, b, c, d] == 0:
                        if que_guidance[a, b, c, d] == 1:
                            mask_cha[a, b, c, d] = 1
    return mask_cha


class Flatten(nn.Module):
    def forward(self, x):
        # x.size(0) retained parameters
        return x.view(x.size(0), -1)

def weights_init_he(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class bce_dice_loss(nn.Module):
    def __init__(self):
        super(bce_dice_loss, self).__init__()
        self.criterion = nn.BCELoss(weight=None, size_average=True, reduce=True)

    def forward(self, y_pred, y_true):
        smooth = 1.
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        # dice_loss
        intersection = (y_true * y_pred).sum()
        dice = (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
        dice_loss = 1.0 - dice
        # binary_loss
        binary_loss = self.criterion(y_pred, y_true)

        return dice_loss + binary_loss


class XPN(nn.Module):
    def __init__(self, in_channels, weights_init=True):
        super().__init__()
        channel = n
        self.in_channels = in_channels
        same_channels = in_channels * channel

        self.conv1_1 = conv2d_3(in_channels, same_channels)
        self.conv1_2 = SubBlock2d(same_channels, same_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = conv2d_3(same_channels, same_channels)
        self.conv2_2 = SubBlock2d(same_channels, same_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = conv2d_3(same_channels, same_channels)
        self.conv3_2 = SubBlock2d(same_channels, same_channels)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = conv2d_3(same_channels, same_channels)
        self.conv4_2 = SubBlock2d(same_channels, same_channels)
        self.pool4 = nn.MaxPool2d(kernel_size=2)


        self.qup1 = nn.Upsample(scale_factor=2)
        self.qconv5_1 = conv2d_3(same_channels * 4, same_channels)
        self.qconv5_2 = SubBlock2d(same_channels, same_channels)

        self.qup2 = nn.Upsample(scale_factor=2)
        self.qconv6_1 = conv2d_3(same_channels * 2, same_channels)
        self.qconv6_2 = SubBlock2d(same_channels, same_channels)

        self.qup3 = nn.Upsample(scale_factor=2)
        self.qconv7_1 = conv2d_3(same_channels * 2, same_channels)
        self.qconv7_2 = SubBlock2d(same_channels, same_channels)

        self.qup4 = nn.Upsample(scale_factor=2)
        self.qconv8_1 = conv2d_3(same_channels * 2, same_channels)
        self.qconv8_2 = SubBlock2d(same_channels, same_channels)

        self.qconv9_1 = conv2d_3(same_channels, same_channels)
        self.qconv9_2 = nn.Sequential(
            nn.Conv2d(same_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax()

        # 设置初始权重
        if weights_init:
            self.apply(weights_init_he)


    def forward(self, input_support_img, input_support_mask, input_query_img):

        channel = n

        convs1_1 = self.conv1_1(input_support_img)
        convs1_2 = self.conv1_2(convs1_1)
        convs1 = torch.add(convs1_1, convs1_2)
        pools1 = self.pool1(convs1)

        convs2_1 = self.conv2_1(pools1)
        convs2_2 = self.conv2_2(convs2_1)
        convs2 = torch.add(convs2_1, convs2_2)
        pools2 = self.pool2(convs2)

        convs3_1 = self.conv3_1(pools2)
        convs3_2 = self.conv3_2(convs3_1)
        convs3 = torch.add(convs3_1, convs3_2)
        pools3 = self.pool3(convs3)

        convs4_1 = self.conv4_1(pools3)
        convs4_2 = self.conv4_2(convs4_1)
        convs4 = torch.add(convs4_1, convs4_2)
        pools4 = self.pool4(convs4)


        convq1_1 = self.conv1_1(input_query_img)
        convq1_2 = self.conv1_2(convq1_1)
        convq1 = torch.add(convq1_1, convq1_2)
        poolq1 = self.pool1(convq1)

        convq2_1 = self.conv2_1(poolq1)
        convq2_2 = self.conv2_2(convq2_1)
        convq2 = torch.add(convq2_1, convq2_2)
        poolq2 = self.pool2(convq2)

        convq3_1 = self.conv3_1(poolq2)
        convq3_2 = self.conv3_2(convq3_1)
        convq3 = torch.add(convq3_1, convq3_2)
        poolq3 = self.pool3(convq3)

        convq4_1 = self.conv4_1(poolq3)
        convq4_2 = self.conv4_2(convq4_1)
        convq4 = torch.add(convq4_1, convq4_2)
        poolq4 = self.pool4(convq4)


        cosine_sim = cs(pools4, poolq4)
        sup_mask_16 = F.interpolate(input_support_mask, pools4.shape[-2:], mode='bilinear',
                                   align_corners=True)
        sup_mask_16_1 = torch.squeeze(sup_mask_16)
        que_mask_16 = gen_mask(cosine_sim, sup_mask_16_1)
        que_mask_16 = torch.unsqueeze(que_mask_16, -3)


        sup_mask_16_repeat = torch.as_tensor(sup_mask_16).repeat(1, channel, 1, 1)
        sup_proto = MaskGAP(pools4, sup_mask_16_repeat)
        sup_proto_repeat = torch.as_tensor(sup_proto).repeat(1, 1, 16, 16)
        que_guidance = cs(sup_proto_repeat, poolq4)
        que_guidance = torch.unsqueeze(que_guidance, -3)


        que_mask_cha16 = qucha(que_mask_16, que_guidance)
        que_fea_16 = que_mask_cha16 * poolq4
        sup_mask_cha16 = qucha(que_mask_16, sup_mask_16)
        sup_mask_cha16_repeat = torch.as_tensor(sup_mask_cha16).repeat(1, channel, 1, 1)
        sup_fea_16_proto = MaskGAP(pools4, sup_mask_cha16_repeat)
        sup_fea_16_proto_repeat = torch.as_tensor(sup_fea_16_proto).repeat(1, 1, 16, 16)
        que_cha_guidance = cs(sup_fea_16_proto_repeat, que_fea_16)
        que_cha_guidance = torch.unsqueeze(que_cha_guidance, -3)


        que_mask_16 = que_mask_16 * sup_proto_repeat
        que_cha_guidance = que_cha_guidance * sup_proto_repeat
        que_guidance = que_guidance * sup_proto_repeat

        que_guidance_all_16 = torch.cat([que_mask_16, que_cha_guidance, que_guidance], dim=1)


        qconcat1 = torch.cat([poolq4, que_guidance_all_16], dim=1)
        qup1 = self.qup1(qconcat1)
        qconv5_1 = self.qconv5_1(qup1)
        qconv5_2 = self.qconv5_2(qconv5_1)
        qconv5 = torch.add(qconv5_1, qconv5_2)

        qconcat2 = torch.cat([qconv5, poolq3], dim=1)
        qup2 = self.qup2(qconcat2)
        qconv6_1 = self.qconv6_1(qup2)
        qconv6_2 = self.qconv6_2(qconv6_1)
        qconv6 = torch.add(qconv6_1, qconv6_2)

        qconcat3 = torch.cat([qconv6, poolq2], dim=1)
        qup3 = self.qup3(qconcat3)
        qconv7_1 = self.qconv7_1(qup3)
        qconv7_2 = self.qconv7_2(qconv7_1)
        qconv7 = torch.add(qconv7_1, qconv7_2)

        qconcat4 = torch.cat([qconv7, poolq1], dim=1)
        qup4 = self.qup4(qconcat4)
        qconv8_1 = self.qconv8_1(qup4)
        qconv8_2 = self.qconv8_2(qconv8_1)
        qconv8 = torch.add(qconv8_1, qconv8_2)

        qconv9_1 = self.qconv9_1(qconv8)
        qconv9_2 = self.qconv9_2(qconv9_1)

        return qconv9_2


def load_train_data():
    support_img_train_path = 'data..'
    support_mask_train_path = 'data..'
    query_img_train_path = 'data..'
    query_mask_train_path = 'data..'

    num_train = num

    support_img_train = np.zeros((num_train, 256, 256))
    support_mask_train = np.zeros((num_train, 256, 256))
    query_img_train = np.zeros((num_train, 256, 256))
    query_mask_train = np.zeros((num_train, 256, 256))

    for m in range(num_train):
        support_img_train[m, :, :] = np.loadtxt(support_img_train_path + '%d.csv' % (m + 1), delimiter=',')
        support_mask_train[m, :, :] = np.loadtxt(support_mask_train_path + '%d.csv' % (m + 1), delimiter=',')
        query_img_train[m, :, :] = np.loadtxt(query_img_train_path + '%d.csv' % (m + 1), delimiter=',')
        query_mask_train[m, :, :] = np.loadtxt(query_mask_train_path + '%d.csv' % (m + 1), delimiter=',')

    support_img_train_final = np.reshape(support_img_train, [-1, 1, 256, 256])
    support_mask_train_final = np.reshape(support_mask_train, [-1, 1, 256, 256])
    query_img_train_final = np.reshape(query_img_train, [-1, 1, 256, 256])
    query_mask_train_final = np.reshape(query_mask_train, [-1, 1, 256, 256])

    return support_img_train_final, support_mask_train_final, query_img_train_final, query_mask_train_final


# testing data
def load_test_data():
    support_img_test_path = 'data..'
    support_mask_test_path = 'data..'
    query_img_test_path = 'data..'
    query_mask_test_path = 'data..'

    num_test = num

    support_img_test = np.zeros((num_test, 256, 256))
    support_mask_test = np.zeros((num_test, 256, 256))
    query_img_test = np.zeros((num_test, 256, 256))
    query_mask_test = np.zeros((num_test, 256, 256))


    for n in range(num_test):
        support_img_test[n, :, :] = np.loadtxt(support_img_test_path + '%d.csv' % (n + 1), delimiter=',')
        support_mask_test[n, :, :] = np.loadtxt(support_mask_test_path + '%d.csv' % (n + 1), delimiter=',')
        query_img_test[n, :, :] = np.loadtxt(query_img_test_path + '%d.csv' % (n + 1), delimiter=',')
        query_mask_test[n, :, :] = np.loadtxt(query_mask_test_path + '%d.csv' % (n + 1), delimiter=',')

    support_img_test_final = np.reshape(support_img_test, [-1, 1, 256, 256])
    support_mask_test_final = np.reshape(support_mask_test, [-1, 1, 256, 256])
    query_img_test_final = np.reshape(query_img_test, [-1, 1, 256, 256])
    query_mask_test_final = np.reshape(query_mask_test, [-1, 1, 256, 256])

    return support_img_test_final, support_mask_test_final, query_img_test_final, query_mask_test_final


def get_evaluation_index_2d(data_result, data_label):
    s1 = s2 = s3 = 0
    tp = fn = tn = fp = 0
    all_index_value = []
    for index in range(data_result.shape[0]):
        data_result_each_image = data_result[index]
        data_label_each_image = data_label[index]
        for j in range(data_result.shape[1]):
            for k in range(data_result.shape[2]):
                if data_result_each_image[j, k] == 1:
                    s1 = s1 + 1
                    if data_label_each_image[j, k] == 1:
                        s2 = s2 + 1
        for m in range(data_result.shape[1]):
            for n in range(data_result.shape[2]):
                if data_label_each_image[m, n] == 1:
                    s3 = s3 + 1
                    if data_label_each_image[m, n] == data_result_each_image[m, n]:
                        tp = tp + 1
                    else:
                        fn = fn + 1
                else:
                    if data_label_each_image[m, n] == data_result_each_image[m, n]:
                        tn = tn + 1
                    else:
                        fp = fp + 1

    # Dice指标
    if (s1 + s3) == 0:
        dice_value = -1
    else:
        dice_value = 2.0 * s2 / (s1 + s3)
    all_index_value.append(dice_value)

    return all_index_value


if __name__ == "__main__":
    path = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    print('loading train_data...')
    support_img_train_data, support_mask_train_data, query_img_train_data, query_mask_train_data = load_train_data()
    support_img_train_data = torch.from_numpy(support_img_train_data)
    support_mask_train_data = torch.from_numpy(support_mask_train_data)
    query_img_train_data = torch.from_numpy(query_img_train_data)
    query_mask_train_data = torch.from_numpy(query_mask_train_data)
    train_dataset = Data.TensorDataset(support_img_train_data, support_mask_train_data, query_img_train_data,
                                       query_mask_train_data)

    print('loading test_data...')
    support_img_test_data, support_mask_test_data, query_img_test_data, query_mask_test_data = load_test_data()
    get_evaluation_index_label = query_mask_test_data
    support_img_test_data = torch.from_numpy(support_img_test_data)
    support_mask_test_data = torch.from_numpy(support_mask_test_data)
    query_img_test_data = torch.from_numpy(query_img_test_data)
    query_mask_test_data = torch.from_numpy(query_mask_test_data)
    test_dataset = Data.TensorDataset(support_img_test_data, support_mask_test_data, query_img_test_data,
                                      query_mask_test_data)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0)

    print('Build model...')
    net = XPN(in_channels=1).to(device)
    criterion_attention = bce_dice_loss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  # 权重衰减
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 权重衰减
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 更新学习率

    all_index_value_list = []
    for epoch in range(60):
        print('epoch: ', epoch + 1)
        print('Start training...')
        net.train()

        for step, (x_support_img, x_support_mask, x_query_img, y_query_mask) in enumerate(train_data_loader):
            x_support_img_train = torch.as_tensor(x_support_img, dtype=torch.float32).to(device)
            x_support_mask_train = torch.as_tensor(x_support_mask, dtype=torch.float32).to(device)
            x_query_img_train = torch.as_tensor(x_query_img, dtype=torch.float32).to(device)
            y_query_mask_train = torch.as_tensor(y_query_mask, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            output = net(x_support_img_train, x_support_mask_train, x_query_img_train)
            loss = criterion_attention(output, y_query_mask_train)
            print('loss:', loss)

            loss.backward()
            optimizer.step()
        scheduler.step(epoch)

        model_path = '{}/model/XPN'.format(path) + str(epoch + 1) + '.pth'
        torch.save(net.state_dict(), model_path)

        print('Start testing...')
        net.load_state_dict(torch.load(model_path))
        net.eval()
        count = 0
        num_test = num
        result_data = np.zeros((num_test, 256, 256))
        with torch.no_grad():
            for index, (x_support_img, x_support_mask, x_query_img, y_query_mask) in enumerate(test_data_loader):
                x_support_img_test = torch.as_tensor(x_support_img, dtype=torch.float32).to(device)
                x_support_mask_test = torch.as_tensor(x_support_mask, dtype=torch.float32).to(device)
                x_query_img_test = torch.as_tensor(x_query_img, dtype=torch.float32).to(device)

                output = net(x_support_img_test, x_support_mask_test, x_query_img_test)
                output = output.squeeze()
                output[output >= 0.5] = 1
                output[output < 0.5] = 0
                iter_data = output.cpu().numpy()

                count = index * batch_size
                for i in range(iter_data.shape[0]):
                    for j in range(iter_data.shape[1]):
                        for k in range(iter_data.shape[2]):
                            result_data[(i + count), j, k] = iter_data[i, j, k]

        savemat('...' % (epoch + 1),
                {'prediction_mask%d' % (epoch + 1): result_data})


        get_evaluation_index_label = np.reshape(get_evaluation_index_label, [-1, 256, 256])
        index_value_list = get_evaluation_index_2d(result_data, get_evaluation_index_label)
        index_value_list = np.reshape(index_value_list, [1, 5])
        all_index_value_list.append(index_value_list)


        five_index_column_name_list = ['Dice']
        if epoch == 0:
            data_five_index = pd.DataFrame(index_value_list, columns=five_index_column_name_list)
            data_five_index.to_csv('...', index=False,
                                   encoding='utf-8')
        else:
            data_five_index = pd.DataFrame(index_value_list)
            data_five_index.to_csv('...', mode='a+',
                                   header=None, index=False,
                                   encoding='utf-8')
