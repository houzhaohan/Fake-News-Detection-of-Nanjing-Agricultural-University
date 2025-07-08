# -*- coding: utf-8 -*-
# 特征工程

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD


class FeatureExtractor:
    def __init__(self, method='tfidf', max_features=5000, n_components=None):
        """ 特征提取器初始化 """
        self.method = method
        self.max_features = max_features
        self.n_components = n_components
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=2,
                max_df=0.95
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=2,
                max_df=0.95
            )
            
        self.svd = None
        if n_components:
            self.svd = TruncatedSVD(n_components=n_components)
    
    def fit_transform(self, X):
        """ 拟合并转换训练数据 """
        features = self.vectorizer.fit_transform(X)
        if self.svd:
            features = self.svd.fit_transform(features)
        return features
    
    def transform(self, X):
        """ 转换测试数据 """
        features = self.vectorizer.transform(X)
        if self.svd:
            features = self.svd.transform(features)
        return features
