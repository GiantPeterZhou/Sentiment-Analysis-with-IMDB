# -*- coding: utf-8 -*-
import os
import re

#正则化预处理文本
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split()]
    return tokenized

#存放数据的文件夹目录
datadict = './aclImdb'

#设置标签
labels = {'pos': 1, 'neg': 0}

#读取数据
def reader(workplace):
    words = []
    label = []
    for l in ('pos', 'neg'):
        path = os.path.join(datadict, workplace, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
                token = tokenizer(text=txt)
            words.append(token)
            label.append(labels[l])
    return words, label


