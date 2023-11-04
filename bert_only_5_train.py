import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
import torch.optim as optim
from utils_5_wavEnc_textTok import Get_data   # 改成我的
from torch.autograd import Variable
from models import getBert_only_5    # 具体到那个函数
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = True    #原本是False，这是用来加速网络的参数，True表示启用

with open('Train_data_w2v2Large_BERTbaseTokenized.pickle', 'rb') as file:
    data = pickle.load(file)

data_save = "BERT_Only_data_5.txt"

parser = argparse.ArgumentParser(description="BERT_Model")
parser.add_argument('--cuda', action='store_false')       # cuda默认开启
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=30, metavar='N')
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=30)   #默认30
parser.add_argument('--lr', type=float, default=1e-5)   #默认1e-3
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--dia_layers', type=int, default=2)
# parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)     # 执行四分类任务
parser.add_argument('--cnn_hangshu', type=int, default=128)  # 这里定义输入图片的行数是80px
args = parser.parse_args()
'''
训练参数：
--cuda: 使用GPU
--batch_size：training batch 
--dropout：
--epochs： training times
GRU参数：
--bid_flag: 
--batch_first:
--dia_layers
--out_class
Padding:
--utt_insize : 务必与谱信息的dim对应。
'''
torch.manual_seed(args.seed)   # 使得全局的随机数都是这个随机数种子，从而保证结果的可复现性


def Train(epoch):
    train_loss = 0
    cnet.train()
    total_samples = 0
    correct_samples = 0
    for batch_idx, (_, target, ids, att) in enumerate(train_loader):
        if args.cuda:
            target, ids, att  = target.cuda(), ids.cuda(), att.cuda()
            # print("已启用GPU训练")

        utt_optim.zero_grad()
        target = target.squeeze()
        cout = cnet(ids, att, args)
        loss = torch.nn.CrossEntropyLoss()(cout, target.long())
        loss.backward()
        utt_optim.step()

        train_loss += loss
        _, predicted = torch.max(cout, 1)
        total_samples += target.size(0)
        correct_samples += (predicted == target).sum().item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Accuracy: {:.2f}%'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval,
                        (correct_samples / total_samples) * 100
            ))
            train_loss = 0


def Test():
    cnet.eval()

    label_pre = []
    label_true = []
    with torch.no_grad():
        for batch_idx, (_, target ,ids, att) in enumerate(test_loader):
            if args.cuda:
                target, ids, att = target.cuda(), ids.cuda(), att.cuda()
            target = Variable(target)
            utt_optim.zero_grad()
            # data_1 = torch.squeeze()
            model_out = cnet(ids, att, args)
            output = torch.argmax(model_out, dim=1)
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
        # print(label_true)
        # print(label_pre)

        # 这里的函数是来自sklearn的，用于计算召回值，f1值，混淆矩阵
        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
    return accuracy_f1, accuracy_recall, label_pre, label_true, CM_test


Final_result = []
Fineal_f1 = []
sum_cm = np.zeros((4, 4))
kf = KFold(n_splits=5)
for index, (train, test) in enumerate(kf.split(data)):
    print(f"开启第{index+1}折的训练：")  # 所以是有10折的训练构成。。。。
    train_loader, test_loader, _, _ = Get_data(data, train, test, args)  # 加上了训练元素的id
    cnet = getBert_only_5()
    if args.cuda:
        cnet = cnet.cuda()

    lr = args.lr
    utt_optimizer = getattr(optim, args.optim)(cnet.parameters(), lr=lr)
    utt_optim = optim.Adam(cnet.parameters(), lr=lr)
    f1 = 0
    recall = 0
    recall_list = []
    f1_list = []
    cm_list = []
    for epoch in range(1, args.epochs + 1):
        Train(epoch)
        accuracy_f1, accuracy_recall, pre_label, true_label, cm = Test()
        recall_list.append(accuracy_recall)
        f1_list.append(accuracy_f1)
        cm_list.append(cm)
        if epoch % 15 == 0:
            lr /= 10
            for param_group in utt_optimizer.param_groups:
                param_group['lr'] = lr

        if (accuracy_f1 > f1 and accuracy_recall > recall):
            name_1 = 'BERT_Only' + str(index) + '.pkl'
            torch.save(cnet.state_dict(), name_1)
            recall = accuracy_recall
            f1 = accuracy_f1
    max_recall = max(recall_list)
    max_f1 = f1_list[recall_list.index(max_recall)]  # 通过在recall列表里检索下标来输出对应的f1数值
    cm = cm_list[recall_list.index(max_recall)]
    sum_cm += cm
    print("成功统计一个混淆矩阵")
    with open(data_save, 'a') as f:
        f.write("第" + str(index + 1) + "折数据：" + "\n" + str(max_recall) + '\n' + str(max_f1) + '\n' + str(cm) + '\n')
        print("输出结果已保存")
with open(data_save, 'a') as f:
    f.write('\n10个最佳混淆矩阵之和是：\n' + str(sum_cm))
    print("最终混淆矩阵：\n",sum_cm)
    print("最终混淆矩阵结果已保存")