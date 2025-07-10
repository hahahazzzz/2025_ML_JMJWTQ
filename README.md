# 推荐系统项目– 电影评分推荐(Movie Recommendation System)
1. 预测用户对某一些电影的打分值
2. 数据集及说明参考：“MovieLens电影推荐数据集 .pdf”
3. 数据集划分，将ratings.csv划分为train set：test set，方式:从ratings.cs中的每一个用户抽取5条以上的数据，形成test set，剩余数据构成train set。模型训练过程中不能使用test set中的数据。
4. 使用合理的评价方法，如RMSE等（由于评级以五星为单位，以半星为增量（0.5星-5.0星），考虑RMSE是否合理）

# 本项目组员：金明俊，王泰乾
# 全部成员：郑睿韬，柳彤，金明俊，王泰乾

[![Python](https://img.shields.io/badge/Python-3.13.2-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3.2-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [配置说明](#配置说明)
- [结果输出](#结果输出)
- [性能评估](#性能评估)
- [可视化分析](#可视化分析)
- [异常值检测](#异常值检测)
- [项目优化](#项目优化)

## 项目概述

这是一个基于LightGBM的系统。项目最初考虑使用SVD矩阵分解进行预测，但经过深入分析发现，用户评分本质上是离散的有序数据，因此将问题重新定义为序数分类任务，这样能更好地保持评分间的顺序关系。系统采用CORAL风格的序数分类算法，结合协同过滤、内容分析、文本挖掘等多维特征工程技术，并提供丰富的可视化分析工具，实现对用户电影评分的精准预测。

### 适用场景

- **电影推荐平台**: 为用户推荐可能喜欢的电影
- **内容分析**: 分析电影质量趋势和观众偏好
- **个性化服务**: 基于用户观影历史提供个性化推荐
- **学术研究**: 推荐系统和机器学习研究的实验平台

### 核心优势

- **算法先进**: 采用LightGBM序数分类技术，保持评分的顺序特性
- **特征丰富**: 融合协同过滤、内容分析、文本挖掘等多种特征
- **可视化完善**: 提供多种图表进行数据分析和结果展示
- **结果管理**: 完整的结果保存和日志记录系统
- **模块化设计**: 易于扩展和维护的代码架构

## 核心特性

### 特征工程

- **协同过滤特征**: 通过SVD矩阵分解，挖掘用户和电影的关联模式（20维隐因子）
- **内容特征**: 从电影的类型、年份等信息中提取结构化特征（20个电影类型）
- **文本特征**: 利用TF-IDF技术分析用户标签偏好（100维TF-IDF特征）
- **用户画像**: 分析用户的评分习惯和偏好倾向（951个用户偏好特征）
- **电影画像**: 评估电影的质量指标和受欢迎程度
- **交叉特征**: 捕捉用户与电影的互动模式（60个协同过滤特征）
- **异常值检测**: 多维度异常值识别和标记，提升数据质量

### 核心算法

- **序数分类**: 采用CORAL风格的多分类器架构，处理评分的有序性
- **LightGBM**: 使用梯度提升决策树，兼顾准确性和速度
- **特征选择**: 自动识别重要特征，提升模型效果
- **参数调优**: 支持网格搜索，找到最佳参数组合

### 评估体系

- **回归指标**: 使用RMSE、MAE、R²等指标衡量预测精度
- **分类指标**: 通过准确率、精确率、召回率、F1-Score评估分类效果
- **分层分析**: 分析不同用户群体、电影类型的模型表现
- **误差诊断**: 识别预测偏差模式，发现异常情况

### 可视化分析

- **预测效果展示**: 散点图和箱线图对比真实值与预测值
- **误差分析**: 误差分布规律、混淆矩阵和用户误差模式
- **特征分析**: 特征重要性排序、相关性热力图和数据分布
- **时间趋势分析**: 评分的时间变化规律
- **用户行为分析**: 用户的行为模式和偏好特征

## 技术架构

本项目采用模块化设计，主要包含以下几个层次：

- **数据层**: MovieLens数据集加载、预处理和质量控制
- **特征工程层**: 协同过滤、内容特征、用户画像、文本特征等多维特征构建
- **模型层**: 基于LightGBM的CORAL风格序数分类器
- **评估层**: 多指标评估和误差分析
- **可视化层**: 预测效果、特征分析等图表生成
- **应用层**: 配置管理和日志系统

## 项目结构

```
2025_ML_Code/
├── README.md                    # 项目文档
├── requirements.txt             # Python依赖包列表
├── config.py                    # 全局配置管理
├── main.py                      # 主程序入口
├── .gitattributes               # Git属性配置
│
├── data/                        # 数据处理模块
│   ├── __init__.py              # 模块初始化
│   ├── data_loader.py           # 数据加载和特征工程
│   ├── data_preprocessing.py    # 数据预处理和异常值检测
│   ├── movies.csv               # 电影信息数据
│   ├── ratings.csv              # 用户评分数据
│   └── tags.csv                 # 用户标签数据
│
├── models/                      # 模型相关模块
│   ├── __init__.py              # 模块初始化
│   ├── train_eval.py            # 模型训练和评估
│   └── model_utils.py           # 模型工具函数
│
├── utils/                       # 工具函数模块
│   ├── __init__.py              # 模块初始化
│   ├── logger.py                # 日志记录工具
│   └── metrics.py               # 评估指标函数
│
├── visualization/               # 可视化模块
│   ├── __init__.py              # 模块初始化
│   ├── basic_plots.py           # 基础图表
│   ├── error_analysis.py        # 误差分析图表
│   ├── feature_plots.py         # 特征分析图表
│   ├── font_config.py           # 中文字体配置
│   └── font_fix.py              # 字体修复工具
│
├── fonts/                       # 字体文件目录
├── output/                      # 输出目录
└── logs/                        # 日志目录
```

### 核心模块说明

| 模块 | 功能描述 | 主要文件 |
|------|----------|----------|
| **config** | 全局配置管理 | `config.py` |
| **data** | 数据加载、预处理、特征工程 | `data_loader.py`, `data_preprocessing.py` |
| **models** | 模型训练、预测、评估 | `train_eval.py`, `model_utils.py` |
| **utils** | 工具函数、日志、评估指标 | `logger.py`, `metrics.py` |
| **visualization** | 可视化分析和图表生成 | `basic_plots.py`, `error_analysis.py`, `feature_plots.py` |


## 安装指南

### 环境要求

- **Python**: 3.13.2（推荐）或3.8+
- **操作系统**: Windows 10+、macOS 10.14+或Ubuntu 18.04+
- **内存**: 至少4GB RAM（推荐8GB+）
- **存储空间**: 预留2GB可用空间

### 依赖包

#### 核心依赖
```
pandas==1.3.5           # 数据处理
numpy==1.21.6           # 数值计算
scipy==1.7.3            # 科学计算
scikit-learn==1.0.2     # 机器学习工具
lightgbm==3.3.2         # 梯度提升模型
```

#### 可视化依赖
```
matplotlib==3.5.1       # 基础绘图
seaborn==0.11.2         # 统计图表
```

#### 工具依赖
```
tqdm==4.62.3            # 进度条
```

#### 数据集结构
```
data/
├── ratings.csv         # 用户评分数据
├── movies.csv          # 电影信息数据
└── tags.csv            # 用户标签数据
```

## 快速开始

### 三步开始预测

```bash
# 第一步：检查数据
ls data/  # 确认能看到 ratings.csv, movies.csv, tags.csv

# 第二步：启动程序
python main.py

# 第三步：查看结果
ls output/  # 浏览生成的预测文件和可视化图表
```

### 输出结果

程序运行完成后会在`output/`目录生成：
- **predictions.csv**: 详细预测结果
- **可视化图表**: 多种分析图表(.png格式)
- **运行日志**: 保存在`logs/`目录，按时间戳命名

## 详细使用说明

### 配置管理

系统提供灵活的配置管理，所有设置都集中在`config.py`文件中。主要配置参数详见[配置说明](#配置说明)章节。

### 数据处理流程

#### 1. 数据加载与清洗

加载MovieLens数据：

```python
from data.data_loader import load_data

# 加载所有数据
ratings, movies, tags, report = load_data(
    enable_preprocessing=True,    # 开启数据清洗
    outlier_strategy='flag'       # 异常值处理策略
)

# 查看数据规模
print(f"评分记录数: {len(ratings):,} 条")
print(f"电影数量: {len(movies):,} 部")
if report:
    print(f"数据质量评分: {report.get('quality_score', 'N/A')}/10")
else:
    print("数据预处理报告: 无")
```

#### 2. 特征工程

系统自动构建多维度特征：

```python
from data.data_loader import (
    create_collaborative_filtering_features,
    create_content_features,
    create_tfidf_tag_features,
    create_user_profile_features,
    create_movie_profile_features
)

# 协同过滤特征
user_f, item_f, user_bias, item_bias = create_collaborative_filtering_features(ratings)

# 内容特征
movies_feats, mlb = create_content_features(movies)

# TF-IDF特征
rat_tag, tag_df = create_tfidf_tag_features(ratings, tags)

# 用户画像
user_stats, user_genre_pref = create_user_profile_features(ratings, movies)

# 电影画像
movie_stats = create_movie_profile_features(ratings)
```

### 模型训练

#### 1. 基础训练

使用序数分类算法进行训练：

```python
from models.train_eval import train_models, predict
from models.model_utils import rating_to_label, label_to_rating

# 准备训练数据
X_train = df[feature_columns].values
y_train = df['rating'].apply(rating_to_label).values

# 启动模型训练
models = train_models(
    X_train, y_train,
    num_classes=10,           # 评分类别数 (0.5-5.0)
    n_estimators=1000,        # 树的数量
    learning_rate=0.05        # 学习率
)

# 生成预测结果
pred_labels = predict(models, X_val)
# 将类别标签转换为评分
from models.model_utils import label_to_rating
pred_ratings = [label_to_rating(label) for label in pred_labels]
```

#### 2. 高级训练选项

自定义训练参数：

```python
# 自定义训练参数
models = train_models(
    X_train, y_train,
    num_classes=10,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,                              # 叶子节点数
    categorical_features=['year_r', 'month_r'], # 分类特征
    verbose=True                                # 显示训练进度
)

# 获取特征重要性
feature_importance = models[0].feature_importances_
top_features = sorted(zip(feature_names, feature_importance), 
                     key=lambda x: x[1], reverse=True)[:20]
print("最重要的20个特征:")
for i, (name, importance) in enumerate(top_features, 1):
    print(f"{i:2d}. {name}: {importance:.4f}")
```

### 评估和可视化

#### 1. 性能评估

```python
from utils.metrics import compute_rmse, evaluate_predictions
from models.train_eval import evaluate_models

# 基础指标
rmse = compute_rmse(true_ratings, pred_ratings)
print(f"RMSE: {rmse:.4f}")

# 详细评估
predictions_df = pd.DataFrame({
    'true_rating': true_ratings,
    'pred_rating': pred_ratings
})
eval_results = evaluate_predictions(predictions_df)
print(f"整体RMSE: {eval_results['overall']['RMSE']:.4f}")
```

#### 2. 可视化分析

```python
from visualization.error_analysis import (
    plot_error_distribution,
    plot_confusion_heatmap,
    plot_user_error_distribution,
    plot_mean_error_per_rating
)
from visualization.feature_plots import (
    plot_top20_feature_importance,
    plot_feature_correlation
)

# 误差分析
plot_error_distribution(predictions_df)
plot_confusion_heatmap(predictions_df)
plot_mean_error_per_rating(predictions_df)

# 特征分析
plot_top20_feature_importance(models, X_train)
plot_feature_correlation(X_train, y_train)
```





## 配置说明

### 核心配置参数

系统的主要配置选项：

| 配置项 | 类型 | 默认值 | 功能说明 |
|--------|------|--------|----------|
| `model_name` | str | "LightGBM_CORAL" | 实验名称 |
| `base_dir` | str | "data" | 数据集存放目录 |
| `latent_dim` | int | 20 | SVD隐因子维度 |
| `tfidf_dim` | int | 100 | TF-IDF最大特征数 |
| `seed` | int | 42 | 随机种子 |
| `num_classes` | int | 10 | 评分类别数 (0.5-5.0，步长0.5) |
| `n_estimators` | int | 1000 | LightGBM树的数量 |
| `learning_rate` | float | 0.05 | 学习率 |
| `num_leaves` | int | 63 | 每棵树的叶子节点数 |



### 性能调优指南

#### 追求更高精度
获得最佳预测效果的配置：
```python
# 精度优先配置
config.latent_dim = 50             # 更多隐因子
config.tfidf_dim = 200             # 更丰富的文本特征
config.n_estimators = 2000         # 更多决策树
config.num_leaves = 127            # 更深的树结构
```

#### 追求更快速度
快速验证或处理大数据集的配置：
```python
# 速度优先配置
config.latent_dim = 10             # 较少隐因子
config.tfidf_dim = 50              # 精简文本特征
config.n_estimators = 500          # 较少树数量
config.learning_rate = 0.1         # 更高学习率
```

### 自定义配置

#### 自定义配置

可以通过继承`Config`类或设置环境变量来自定义配置参数。详细示例请参考项目代码中的配置文件。



## 可视化分析

### 模块功能

可视化模块提供全面的数据分析和结果展示功能：

- **预测效果**: 箱线图对比、评分分布直方图
- **误差分析**: 误差分布、混淆矩阵、用户误差分析
- **特征分析**: 特征重要性排序、相关性热力图
- **中文支持**: 自动检测和配置中文字体

### 使用示例

```python
from visualization import *

# 准备预测结果DataFrame
output_df = pd.DataFrame({
    'true_rating': y_true,
    'pred_rating': y_pred,
    'error': y_pred - y_true
})

# 生成各类图表
plot_boxplot_true_vs_pred(output_df)
plot_error_distribution(output_df)
plot_confusion_heatmap(output_df)
plot_top20_feature_importance(model, feature_names)
```



### 自定义配置

支持自定义图表样式和中文字体设置。系统会自动检测并配置合适的中文字体。

## 结果输出

### 输出文件组织

系统会将所有结果保存到output目录：
```
output/
├── predictions.csv                    # 详细预测结果
├── boxplot_true_vs_pred.png          # 真实值vs预测值箱线图
├── predicted_rating_hist.png         # 预测评分分布直方图
├── prediction_error_hist.png         # 预测误差分布直方图
├── mean_error_per_rating.png         # 各评分等级平均误差
├── confusion_heatmap.png             # 混淆矩阵热力图
├── top20_feature_importance.png      # Top20特征重要性
├── feature_correlation.png           # 特征相关性热力图
├── user_error_distribution.png       # 用户误差分布
├── rmse_per_rating_level.png         # 各评分等级RMSE
├── error_vs_popularity_line.png      # 误差与流行度关系
└── feature_distributions.png         # 特征分布图

logs/                                 # 运行日志
└── run_YYYYMMDD_HHMMSS.log          # 按时间戳命名的运行日志
```

## 性能评估

### 评估体系

项目使用多种评估指标衡量模型表现：

#### 主要指标
- **RMSE**: 均方根误差，主要评估指标，越小越好
- **MAE**: 平均绝对误差，直观反映预测偏差
- **准确率**: 完全匹配预测的比例

#### 辅助指标
- **精确率/召回率**: 评估不同评分等级的预测质量
- **F1分数**: 平衡精确率和召回率
- **NDCG**: 考虑排序位置的推荐质量评估

### 性能基准测试

#### 性能基准

在MovieLens数据集上的主要性能指标：

- **RMSE**: 0.8166 (均方根误差，越小越好)
- **数据规模**: 100,836条评分记录
- **用户数量**: 610个用户
- **电影数量**: 9,742部电影
- **特征维度**: 1,141个特征
- **训练时间**: ~5.3分钟 (317秒，标准配置)

#### 数据质量与异常值检测

- **异常值比例**: 11.13% (11,223条记录)
  - **行为异常**: 9.07% (用户评分行为异常)
  - **时间异常**: 0.0% (时间异常检测已禁用)
  - **多维异常**: 11.13% (多维特征空间异常)

#### 性能特点

- **高评分预测**: 4-5星评分预测相对困难，因为样本集中且差异细微
- **低评分预测**: 1-2星评分样本较少但预测相对容易
- **用户差异**: 活跃用户(评分多)的预测准确性明显高于新用户
- **异常值处理**: 通过多维异常检测提升数据质量，改善模型性能

## 异常值检测

### 检测策略

项目实现了多维度的异常值检测系统，用于识别和标记可能影响模型性能的异常数据：

#### 1. 用户行为异常检测
- **评分数量异常**: 使用Z-score检测评分数量异常的用户（阈值：4.0）
- **评分方差异常**: 识别评分方差过小的用户（标准差<0.01且评分数≥10）
- **极端评分用户**: 检测评分范围过窄的高频用户

#### 2. 时间模式异常检测
- **短时间间隔**: 检测10秒内连续评分的异常行为
- **异常比例阈值**: 短间隔比例>95%且总间隔数≥10的用户

#### 3. 多维特征异常检测
- **IsolationForest算法**: 使用隔离森林检测多维特征空间中的异常点
- **参数配置**: contamination=0.1, n_estimators=200
- **最小评分要求**: 仅对评分数≥20的用户进行检测

### 检测结果

基于最新实验的异常值检测结果：

| 异常类型 | 检测数量 | 占比 | 说明 |
|----------|----------|------|------|
| 行为异常 | 9,148 | 9.07% | 用户评分行为模式异常 |
| 时间异常 | 0 | 0.0% | 时间异常检测已禁用 |
| 多维异常 | 11,223 | 11.13% | 多维特征空间异常 |
| **总计** | **11,223** | **11.13%** | **综合异常检测结果** |

### 优化效果

## 项目优化

### 已完成的优化

#### 1. 异常值检测优化
- **参数精调**: 优化各类异常检测的阈值参数
- **算法改进**: 改进IsolationForest的参数配置
- **检测精度**: 显著降低误报率，提升检测准确性

#### 2. 特征工程优化
- **特征维度**: 构建1,141维综合特征向量
- **特征类型**: 包含协同过滤、内容、文本、用户画像等多类特征
- **特征选择**: 自动识别重要特征，提升模型效果

#### 3. 模型性能优化
- **CORAL算法**: 采用序数分类算法保持评分顺序特性
- **LightGBM**: 使用高效的梯度提升框架
- **参数调优**: 优化树的数量、学习率等关键参数

---

*最后更新时间: 2025年7月*
