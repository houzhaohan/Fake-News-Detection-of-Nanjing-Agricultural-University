# Fake-News-Detection-of-Nanjing-Agricultural-University
Fake News Detection of Nanjing Agricultural University 关于南京农业大学的虚假新闻检测  
Python语言-课程作业  
大三下  

# 关于南京农业大学的虚假新闻检测

## 一、实验目的

​	本项目构建了基于机器学习的自动化虚假新闻检测系统，旨在通过端到端的文本二分类技术解决网络信息真实性验证问题。

​	核心目标聚焦于开发完整的自然语言处理程序，从原始数据清洗、特征工程到模型训练与评估，形成可复用的分类框架。研究重点在于验证不同机器学习技术（随机森林、支持向量机、逻辑回归、朴素贝叶斯）对有关南京农业大学虚假新闻二分类效果的差异化影响，同时系统对比传统机器学习模型在文本二分类任务中的性能表现。通过分层抽样和交叉验证方法建立可靠的评估体系，最终形成兼具实用性和扩展性的虚假新闻检测解决方案。

## 二、数据预处理与特征工程

### 2.1数据集介绍

​	使用crawler.py脚本从南京农业大学新闻网爬取新闻标题，保存为news.txt文件，共计1386条真实新闻。然后把news.txt文件转换成结构化数据news.csv文件（第一列是行号，第二列是新闻文本，第三列是0，0代表真实新闻）。

​	使用fake_news.py生成虚假新闻文本，从词库中抽取词替换掉真实新闻的30%构成虚假新闻，共计1385条虚假新闻。虚假新闻保存到news.csv后。

​	news.csv共计2771条新闻数据。

### 2.2数据预处理

​	使用data_preprocessing.py进行数据预处理原始文本通过正则表达式清洗流程完成标准化，该步骤可有效消除网页抓取残留的格式符号，确保文本仅包含有效语义内容。

​	采用Jieba分词器的精确模式进行词语切分，为后续特征工程做铺垫。

​	使用分层抽样划分训练集与测试集（8:2）。分层抽样确保训练集（2,217条）与测试集（554条）的类别比例与原始数据集完全一致，避免因随机划分导致的分布偏移问题。

### 2.3特征工程

​	feature_extraction.py是特征工程脚本，特征工程围绕文本向量化与特征降维两个核心阶段展开。

​	采用TF-IDF（词频-逆文档频率）算法对分词后的文本进行数学建模，通过统计词语在文档中的出现频率与全局分布特征，构建能够反映词语重要性的高维稀疏矩阵。在向量化阶段设置最大特征数阈值（默认5000维），配合最小文档频率（min_df=2）和最大文档频率（max_df=0.95）参数，有效过滤低频噪声词与泛化高频词。

​	经过向量空间映射后的文本特征，根据配置参数选择性进行潜在语义分析降维。当启用TruncatedSVD组件时，系统将高维稀疏向量投影至300维左右的稠密语义空间，既保留关键语义特征又显著降低计算复杂度。

## 三、模型训练与模型选择

### 3.1模型训练模块

​	model_training.py统一接口封装不同机器学习模型（随机森林、支持向量机、逻辑回归、朴素贝叶斯）的初始化、训练及预测逻辑，为后续模型评估和选择提供基础能力。该模块使得主程序能够灵活切换不同模型类型，同时支持超参数调优，显著提升了代码的可维护性和扩展性。其核心作用是将模型训练的具体实现细节封装起来，让上层逻辑（如模型选择、评估）无需关注不同模型的训练差异，从而降低了系统的耦合度。

### 3.2模型选择模块

​	model_evaluation.py通过定义静态类ModelEvaluator，提供了一套完整的模型评估工具，主要用于量化分析模型预测效果、生成可视化结果并输出详细评估报告，在模型选择与优化过程中起到关键作用。该模块计算并返回准确率（accuracy）、精确率（precision）、召回率（recall）、F1分数（f1）等基础分类指标；若提供了预测概率，还会额外计算AUC（曲线下面积）和对数损失（log_loss），这些指标从不同维度反映模型的分类能力。并且画出混淆矩阵。

## 四、实验结果与用例测试

### 4.1实验结果

​	运行main.py可以完整调用所有模块，进行有关南京农业大学虚假信息检测。

![image](https://github.com/user-attachments/assets/84dad734-d815-498b-be40-3b114f5867a0)

​	main.py首先会运行data_preprocessing.py进行数据预处理，然后运行feature_extraction.py进行特征工程。然后运行model_training.py和model_evaluation.py进行模型训练和选择。

​	首先是随机森林：

![image](https://github.com/user-attachments/assets/23c5c703-307f-439b-ab58-15fadaba8645)

![image](https://github.com/user-attachments/assets/a6eded4c-38e1-4545-af4c-860599461ce3)

​	其次是支持向量机：

![image](https://github.com/user-attachments/assets/2549e532-89d5-43ea-add4-75564e41180b)

![image](https://github.com/user-attachments/assets/78ec3ac1-c285-48c1-9a07-24c40b7e210a)

​	其次是逻辑回归：

![image](https://github.com/user-attachments/assets/d292a4e8-be9f-429c-9e55-0d71b550a52a)

![image](https://github.com/user-attachments/assets/df590826-c215-4662-911a-776618cc7ef0)

​	最后是朴素贝叶斯：

![image](https://github.com/user-attachments/assets/9414e90d-f201-495d-aa63-d90c6a27269f)

![image](https://github.com/user-attachments/assets/6a20e02e-37c7-4307-acc1-5fd96b08eda6)

​	根据F1分数指标判断出最优模型：

![image](https://github.com/user-attachments/assets/4e3f004f-dcec-4e41-90c9-1d693d0444d4)

​	运行过程中会把文本向量参数和最优模型参数分别保存为feature_extractor.joblib和best_model.joblib。

### 4.2用例测试

​	运行predict.py可以进行用例测试，模型参数使用训练好的best_model.joblib。

![image](https://github.com/user-attachments/assets/9178eb5f-478e-4468-8a6e-3c42a42b1788)

![image](https://github.com/user-attachments/assets/c0bc4063-e065-4ff6-bf94-4221660336b5)

## 五、实验结论

​	本次项目围绕“有关南京农业大学的虚假新闻检测”主题，通过四种机器学习方法成功构建了高效的虚假新闻识别模型。实验基于规范的数据预处理（正则清洗、Jieba分词、分层抽样划分数据集）、科学的特征工程（TF-IDF向量化结合TruncatedSVD降维）及严谨的多模型对比评估（随机森林、SVM、逻辑回归、朴素贝叶斯），最终筛选出最优模型并保存关键参数（feature_extractor.joblib与best_model.joblib）。模型在测试集上表现出可靠的分类性能，通过准确率、F1值、AUC等指标及混淆矩阵验证了其对真实与虚假新闻的有效区分能力。用例测试（predict.py）进一步表明，模型能够准确识别南京农业大学相关的真实与虚假新闻，具备实际应用价值。项目成功实现了南京农业大学相关虚假新闻的自动化检测目标，为同类文本分类任务提供了可复用的技术方案与实践参考。
