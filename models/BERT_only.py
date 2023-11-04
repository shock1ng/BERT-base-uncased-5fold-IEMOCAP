from torch import nn
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import pandas as pd
import re

# 指定本地模型路径 这两个是服务bert用的
model_path = "bert-base-uncased"

# 加载tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(model_path)  #（1）如果模型内的不好用，就用模型外的
MAX_LEN = 128  # token后最长128
# 读取CSV文件
data_frame = pd.read_csv("IEMOCAP_sentence_trans.csv")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class getBert_only(nn.Module):
    def __init__(self):
        super(getBert_only, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(768, 100)     # large模型出来的是1024的，而不是768
        self.norm = nn.BatchNorm1d(100)   #再加一个归一化层看看效果
        self.activ = nn.ReLU()
        self.linear2 = nn.Linear(100, 4)
        self.FC = nn.Linear(768,4)

    def forward(self, spec_id, args):
        input_ids_list = []
        attention_mask_list = []
        for item in spec_id:
            # 查询给定编号对应的文本和标签
            result = data_frame[data_frame['id'] == item]
            if len(result) == 0:
                return None, None
            text = result['transcription'].values[0]
            # 使用正则表达式去除标点符号
            text = re.sub(r'[^\w\s]', '', text)

            # 对输入文本进行tokenization并添加特殊token
            encoding = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN,
                                      return_tensors='pt')

            input_ids = encoding.get('input_ids').squeeze().to(device)  # 把[1,128]变成[128]
            attention_mask = encoding.get('attention_mask').squeeze().to(device)   # 把[1,128]变成[128]

            # 把两位append进列表中，等待下一步的堆叠
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        input_ids = torch.cat(input_ids_list, dim=0).view(args.batch_size, MAX_LEN)  #堆叠成[batch,MAX_LEN]
        attention_mask = torch.cat(attention_mask_list, dim=0).view(args.batch_size, MAX_LEN)   #堆叠成[batch,MAX_LEN]

        output = self.bert(input_ids, attention_mask)  #改换思路，把矩阵丢进bert
        x = output.pooler_output  # 拿取pooler层的输出 ,large是[batch,1024]和base是[batch,768]

        if args.cuda:
            x = x.cuda()

        x = self.FC(x)
        return x

class getBert_only_5(nn.Module):
    def __init__(self):
        super(getBert_only_5, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(768, 100)     # large模型出来的是1024的，而不是768
        self.norm = nn.BatchNorm1d(100)   #再加一个归一化层看看效果
        self.activ = nn.ReLU()
        self.linear2 = nn.Linear(100, 4)
        self.FC = nn.Linear(768,4)

    def forward(self, input_ids , attention_mask, args):
        output = self.bert(input_ids, attention_mask)  #改换思路，把矩阵丢进bert
        x = output.pooler_output  # 拿取pooler层的输出 ,large是[batch,1024]和base是[batch,768]

        if args.cuda:
            x = x.cuda()
        x = self.FC(x)
        return x

#以下是拿取文本矩阵的废稿
'''
def BERT_by_id(data_frame, target_id, args):
    output = []
    for item in target_id:
        # 查询给定编号对应的文本和标签
        result = data_frame[data_frame['id'] == item]
        if len(result) == 0:
            return None, None
        text = result['transcription'].values[0]
        # 使用正则表达式去除标点符号
        text = re.sub(r'[^\w\s]', '', text)

        # 步骤3: 对输入文本进行tokenization并添加特殊token
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

        input_ids = input_ids.cuda()     # 自从在上面把Bmodel和这个放在cuda上之后，训练速度飞速提升！！！显存占用多了600兆

        # 步骤4: 前向传播获得BERT模型的输出
        tmp_out = Bmodel(input_ids)

        # 步骤5: 获取输出向量
        tmp_output_vector = tmp_out[1]  # 取得句子的CLS向量, 是张量类型

        # 输出结果
        tmp_out = tmp_output_vector.tolist()  # 这里转换为列表，因为张量我操作不来，这里的形状是（1，768）
        # tmp_out = z_score_scaling(tmp_out)  # 进去归一化 二选一，这俩效果差不多
        # tmp_out = min_max_scaling(tmp_out)  # 进去归一化

        output.append(tmp_out)  # 逐一加入到空列表中

    output = np.stack(output, axis=0)  # batch个（1，768）堆叠变成（128，1，768）
    output = torch.tensor(output)  # 把列表变成张量数据类型
    # print(output.shape)                 # torch.Size([128, 1, 768])

    return output  # torch.Size([128, 1, 768])
'''


'''
class CNN_BERT(nn.Module):
    def __init__(self):
        super(CNN_BERT, self).__init__()
        # 定义模型结构
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 100)  # 输出100个元素的一维矩阵，bert那边也出100个(原本是768，线性成100)，两者占比一半一半

        )
        self.bert = getBert_only().cuda()
        self.linear = nn.Linear(100,4)

    def forward(self, x, spec_id, args):
        spec = self.model(x)          # spec图的[128,100]
        text_vt = self.bert(spec_id, args)  # bert出来的[128,100]
        #操作前先转列表
        spec = spec.tolist()
        text_vt = text_vt.tolist()
        sum_all = []
        for i in range(len(spec)):
            spec_dot_cText = np.dot(spec[i], cyclic_matrix(text_vt[i]))
            text_dot_cSpec = np.dot(text_vt[i], cyclic_matrix(spec[i]))
            sum = spec_dot_cText + text_dot_cSpec
            sum_all.append(sum)

        sum_all = torch.Tensor(sum_all)
        # print(sum_all.shape)    #  确认是torch.Size([128, 100])没错
        if args.cuda:
            x = sum_all.cuda()
        x = self.linear(x)

        return x

'''