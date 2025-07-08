# -*- coding: utf-8 -*-
# 主函数

import pandas as pd
import joblib
from data_preprocessing import clean_text, segment_text
from feature_extraction import FeatureExtractor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator


def main():
    # 加载数据
    print("加载数据...")
    df = pd.read_csv('news.csv', encoding='utf-8')
    df.columns = ['index', 'text', 'label']
    
    # 数据预处理
    print("数据预处理...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['segmented_text'] = df['cleaned_text'].apply(segment_text)
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df['segmented_text'], 
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # 特征提取
    print("特征提取...")
    # 为朴素贝叶斯创建单独的特征提取器（使用计数向量化）
    nb_feature_extractor = FeatureExtractor(
        method='count',
        max_features=5000,
        n_components=None  # 朴素贝叶斯不使用LSA降维
    )
    
    # 为其他模型创建TF-IDF特征提取器
    tfidf_feature_extractor = FeatureExtractor(
        method='tfidf',
        max_features=5000,
        n_components=300  # 使用LSA降维
    )
    
    # 提取特征
    X_train_nb = nb_feature_extractor.fit_transform(X_train)
    X_test_nb = nb_feature_extractor.transform(X_test)
    
    X_train_tfidf = tfidf_feature_extractor.fit_transform(X_train)
    X_test_tfidf = tfidf_feature_extractor.transform(X_test)
    
    # 模型训练
    print("模型训练...")
    models = {
        'rf': (ModelTrainer(model_type='rf'), (X_train_tfidf, X_test_tfidf)),
        'svm': (ModelTrainer(model_type='svm'), (X_train_tfidf, X_test_tfidf)),
        'lr': (ModelTrainer(model_type='lr'), (X_train_tfidf, X_test_tfidf)),
        'nb': (ModelTrainer(model_type='nb'), (X_train_nb, X_test_nb))
    }
    results = {}
    for name, (model, (X_train_feat, X_test_feat)) in models.items():
        print(f"\n训练 {name} 模型...")
        model.train(X_train_feat, y_train)
        
        # 预测
        y_pred = model.predict(X_test_feat)
        y_prob = model.predict_proba(X_test_feat)

        # 评估
        print(f"\n{name} 模型评估结果:")
        results[name] = ModelEvaluator.evaluate(y_test, y_pred, y_prob)
        ModelEvaluator.print_evaluation(y_test, y_pred, y_prob, model_name=name)
    
    # 选择最佳模型
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    print(f"\n最佳模型是: {best_model_name}")
    
    # 保存最佳模型和特征提取器
    print("保存模型...")
    best_model, (_, _) = models[best_model_name]
    feature_extractor = tfidf_feature_extractor if best_model_name != 'nb' else nb_feature_extractor
    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(feature_extractor, 'feature_extractor.joblib')


if __name__ == "__main__":
    main()
