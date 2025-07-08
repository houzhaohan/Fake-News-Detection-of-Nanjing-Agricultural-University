# -*- coding: utf-8 -*-
# 模型训练

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


class ModelTrainer:
    def __init__(self, model_type='rf', **kwargs):
        """
        模型训练器初始化
        model_type: 'rf' (随机森林), 'svm' (支持向量机), 'lr' (逻辑回归), 'nb' (朴素贝叶斯)
        """
        self.model_type = model_type
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                **kwargs
            )
        elif model_type == 'svm':
            self.model = SVC(
                probability=True,
                random_state=42,
                **kwargs
            )
        elif model_type == 'lr':
            self.model = LogisticRegression(
                random_state=42,
                **kwargs
            )
        elif model_type == 'nb':
            self.model = MultinomialNB(**kwargs)
        else:
            raise ValueError(f"不支持模型类型: {model_type}")
    
    def train(self, X, y):
        """ 训练模型 """
        self.model.fit(X, y)
        
    def predict(self, X):
        """ 预测 """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """ 预测概率 """
        return self.model.predict_proba(X)
