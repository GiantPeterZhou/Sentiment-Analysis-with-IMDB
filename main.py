# -*- coding: utf-8 -*-
from preprocessing import reader

#生成训练、测试数据
x_train, y_train = reader('train')
x_test, y_test = reader('test')

from gensim.models.word2vec import Word2Vec

#第一种训练方法
#建立模型，默认的min_count值为5
#size参数主要是用来设置神经网络的层数，Word2Vec 中的默认值是设置为100层
#model = Word2Vec(x_train, size = 300, min_count=1)
#这里我们调用Word2Vec创建模型实际上会对数据执行两次迭代操作
#第一轮操作会统计词频来构建内部的词典数结构，第二轮操作会进行神经网络训练

#第二种训练方法
#也可以分两次迭代
#先建立一个空的模型，不训练
model = Word2Vec(size = 300, min_count=1)
#遍历一次语料库建立词典
model.build_vocab(x_train)
#第二次遍历语料库建立神经网络模型，数据可以与第一次不同
model.train(x_train, total_examples = model.corpus_count, epochs= model.iter)

#获取每句句子中，所有词向量的平均值
import numpy as np
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

#标准化训练数据
from sklearn.preprocessing import scale
train_vec = np.concatenate([buildWordVector(z, 300) for z in x_train])
train_vec = scale(train_vec)

#增量训练，可以先更新词汇库，再完成训练
model.build_vocab(x_test, update=True)
model.train(x_test, total_examples= model.corpus_count, epochs = model.iter)

#标准化测试数据
test_vec = np.concatenate([buildWordVector(z, 300) for z in x_test])
test_vec = scale(test_vec)

#建模，使用随机梯度下降法作为分类器算法
from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier(loss = 'log')
SGD.fit(train_vec, y_train)
accuracy = SGD.score(test_vec, y_test)
print('Test Accuracy: %.2f'%accuracy)