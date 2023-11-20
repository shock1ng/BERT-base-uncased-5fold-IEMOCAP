# BERT-base-uncased
这里是纯BERT的部分，调用BERT对文本进行加token再进网络，进行IEMOCAP的情感分类。
data_pp的运行需要：
1、安装librosa；
2、必须从Hugging face下载了BERT-base-uncased和wav2vec2的全部内容，目的是全部打包好，方便之后的调取训练
3、必须把IEMOCAP的整个文件解压放在某目录，然后把这个目录和代码里的目录做替换

BERT_only.py 需要放进名为：models的文件夹内才可以运行
最终的训练结果会在本地保存一个txt，训练F1精度大概在70~72%左右
