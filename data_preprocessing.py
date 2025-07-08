# -*- coding: utf-8 -*-
# 数据预处理

import pandas as pd
import jieba
import re
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """ 加载数据并进行基本的预处理 """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 假设列名为：index, text, label
    df.columns = ['index', 'text', 'label']
    return df


def clean_text(text):
    """ 清理文本数据 """
    # 移除特殊字符和数字
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', str(text))
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def segment_text(text):
    """ 对文本进行分词 """
    words = jieba.lcut(text)
    return ' '.join(words)


def prepare_data(df):
    """ 准备数据集 """
    # 清理文本
    df['cleaned_text'] = df['text'].apply(clean_text)
    # 分词
    df['segmented_text'] = df['cleaned_text'].apply(segment_text)
    
    # 分割数据集
    X = df['segmented_text']
    y = df['label']
    
    # 使用分层抽样，保持标签分布
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
