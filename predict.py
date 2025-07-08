# -*- coding: utf-8 -*-
# 给定文本预测真实性

import joblib
from data_preprocessing import clean_text, segment_text


def load_models():
    """ 加载保存的模型和特征提取器 """
    try:
        model = joblib.load('best_model.joblib')
        feature_extractor = joblib.load('feature_extractor.joblib')
        return model, feature_extractor
    except FileNotFoundError:
        raise Exception("模型文件未找到。请先运行train_model.py训练模型。")


def predict_news(text, model, feature_extractor):
    """ 预测单条新闻是否为虚假新闻 """
    # 预处理文本
    cleaned_text = clean_text(text)
    segmented_text = segment_text(cleaned_text)
    
    # 特征提取
    features = feature_extractor.transform([segmented_text])
    
    # 预测
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        'is_fake': bool(prediction),
        'probability': probability[1],  # 虚假新闻的概率
        'confidence': max(probability)
    }


def main():
    # 加载模型
    print("加载模型...")
    model, feature_extractor = load_models()
    
    while True:
        # 获取用户输入
        print("\n请输入要检测的新闻文本（输入 'q' 退出）：")
        text = input().strip()
        
        if text.lower() == 'q':
            break
            
        if not text:
            print("请输入有效的新闻文本！")
            continue
        
        try:
            # 进行预测
            result = predict_news(text, model, feature_extractor)
            
            # 输出结果
            print("\n预测结果：")
            print(f"新闻类型: {'虚假新闻' if result['is_fake'] else '真实新闻'}")
            print(f"预测置信度: {result['confidence']*100:.2f}%")
            
            if result['is_fake']:
                if result['probability'] > 0.9:
                    print("警告：这很可能是一条虚假新闻！")
                elif result['probability'] > 0.7:
                    print("提示：这条新闻的真实性值得怀疑。")
                else:
                    print("提示：这条新闻可能包含一些不准确信息。")
        
        except Exception as e:
            print(f"预测出错：{str(e)}")


if __name__ == "__main__":
    main()
