# -*- coding: utf-8 -*-
# 模型选择

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred, y_prob=None):
        """ 评估模型性能 """
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            # 如果有概率预测，计算AUC和对数损失
            from sklearn.metrics import roc_auc_score, log_loss
            results['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            results['log_loss'] = log_loss(y_true, y_prob)
        
        return results

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name):
        """ 绘制混淆矩阵图 """
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

        plt.figure(figsize=(8, 6))
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # 使用seaborn绘制热力图
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['真实新闻', '虚假新闻'],
                   yticklabels=['真实新闻', '虚假新闻'])
        
        plt.title(f'{model_name}模型的混淆矩阵')
        plt.ylabel('实际类别')
        plt.xlabel('预测类别')
        
        # 创建results目录（如果不存在）
        os.makedirs('results', exist_ok=True)
        
        # 保存图片
        plt.savefig(f'results/confusion_matrix_{model_name}.png', 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()
    
    @staticmethod
    def print_evaluation(y_true, y_pred, y_prob=None, model_name=None):
        """ 打印详细的评估报告 """
        print("\n分类报告:")
        print(classification_report(y_true, y_pred))
        
        print("\n混淆矩阵:")
        conf_matrix = confusion_matrix(y_true, y_pred)
        print(conf_matrix)
        
        # 绘制混淆矩阵图
        if model_name:
            ModelEvaluator.plot_confusion_matrix(y_true, y_pred, model_name)
            print(f"\n混淆矩阵图已保存到 results/confusion_matrix_{model_name}.png")
        
        if y_prob is not None:
            print("\n预测概率统计:")
            print(f"平均预测概率: {np.mean(y_prob[:, 1]):.4f}")
            print(f"最大预测概率: {np.max(y_prob[:, 1]):.4f}")
            print(f"最小预测概率: {np.min(y_prob[:, 1]):.4f}")
            
            # 计算AUC和对数损失
            from sklearn.metrics import roc_auc_score, log_loss
            print(f"\nAUC: {roc_auc_score(y_true, y_prob[:, 1]):.4f}")
            print(f"Log Loss: {log_loss(y_true, y_prob):.4f}")
