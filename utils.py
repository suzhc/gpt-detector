import pandas as pd
import numpy as np


def read_train_data():

    train_data = pd.read_csv('data/ChatGPT生成文本检测器公开数据-更新/train.csv')
    train_data['content'] = train_data['content'].apply(lambda x: x[1:-1].strip().replace('\n', ' \n '))
    train_data['content'] = train_data['content'].apply(lambda x: x.split(' '))
    train_data['content'] = train_data['content'].apply(lambda x: [i for i in x if i != ''])
    
    return train_data

def read_test_data():

    test_data = pd.read_csv('data/ChatGPT生成文本检测器公开数据-更新/test.csv')
    test_data['content'] = test_data['content'].apply(lambda x: x[1:-1].strip().replace('\n', ' \n '))
    test_data['content'] = test_data['content'].apply(lambda x: x.split(' '))
    test_data['content'] = test_data['content'].apply(lambda x: [i for i in x if i != ''])
    
    return test_data