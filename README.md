# 推荐系统项目– 电影评分推荐(Movie Recommendation System)
1. 预测用户对某一些电影的打分值
2. 数据集及说明参考：“MovieLens电影推荐数据集 .pdf”
3. 数据集划分，将ratings.csv划分为train set：test set，方式:从ratings.cs中的每一个用户抽取5条以上的数据，形成test set，剩余数据构成train set。模型训练过程中不能使用test set中的数据。
4. 使用合理的评价方法，如RMSE等（由于评级以五星为单位，以半星为增量（0.5星-5.0星），考虑RMSE是否合理）

# 本项目组员：金明俊，王泰乾

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Development-orange.svg)]()

这是一个基于LightGBM的系统。项目最初考虑使用SVD矩阵分解进行预测，但经过深入分析发现，用户评分本质上是离散的有序数据，因此将问题重新定义为序数分类任务，这样能更好地保持评分间的顺序关系。系统采用CORAL风格的序数分类算法，结合协同过滤、内容分析、文本挖掘等多维特征工程技术，并提供丰富的可视化分析工具，实现对用户电影评分的精准预测。

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [配置说明](#配置说明)
- [实验管理](#实验管理)
- [性能评估](#性能评估)
- [可视化分析](#可视化分析)

## 项目概述

本项目是一个电影评分预测系统，旨在准确预测用户对电影的评分偏好。采用序数分类技术，将评分预测转化为多个二分类问题，更好地保持评分之间的顺序关系。

### 适用场景

- **电影推荐平台**: 为用户推荐可能喜欢的电影
- **内容分析**: 分析电影质量趋势和观众偏好
- **个性化服务**: 基于用户观影历史提供个性化推荐
- **学术研究**: 推荐系统和机器学习研究的实验平台

### 核心优势

- **算法先进**: 采用LightGBM序数分类技术，保持评分的顺序特性
- **特征丰富**: 融合协同过滤、内容分析、文本挖掘等多种特征
- **可视化完善**: 提供多种图表进行数据分析和结果展示
- **实验管理**: 完整的实验记录和追踪系统
- **模块化设计**: 易于扩展和维护的代码架构

## 核心特性

### 特征工程

- **协同过滤特征**: 通过SVD矩阵分解，挖掘用户和电影的关联模式
- **内容特征**: 从电影的类型、年份等信息中提取结构化特征
- **文本特征**: 利用TF-IDF技术分析用户标签偏好
- **用户画像**: 分析用户的评分习惯和偏好倾向
- **电影画像**: 评估电影的质量指标和受欢迎程度
- **交叉特征**: 捕捉用户与电影的互动模式

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
- **应用层**: 实验管理、配置管理和日志系统

## 项目结构

```
2025_ML_Code/
├── README.md                    # 项目文档
├── requirements.txt             # Python依赖包列表
├── config.py                    # 全局配置管理
├── main.py                      # 主程序入口
│
├── data/                        # 数据处理模块
│   ├── __init__.py              # 模块初始化
│   ├── data_loader.py           # 数据加载和特征工程
│   ├── data_preprocessing.py    # 数据预处理和清洗
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
│   └── feature_plots.py         # 特征分析图表
│
├── experiments/                 # 实验管理模块
│   ├── __init__.py              # 模块初始化
│   ├── experiment.py            # 实验管理类
│   └── [实验记录目录]/           # 输出：各次实验的结果
│       ├── config.json          # 实验配置
│       ├── results.json         # 实验结果
│       ├── predictions.csv      # 预测结果
│       ├── plots/               # 可视化图表
│       ├── models/              # 训练模型
│       └── logs/                # 实验日志
│
├── output/                      # 输出：输出目录
│   ├── predictions.csv          # 最新预测结果
│   └── *.png                    # 生成的图表文件
│
└── logs/                        # 输出：日志目录
    └── *.log                    # 运行日志文件
```

### 核心模块说明

| 模块 | 功能描述 | 主要文件 |
|------|----------|----------|
| **config** | 全局配置管理 | `config.py` |
| **data** | 数据加载、预处理、特征工程 | `data_loader.py`, `data_preprocessing.py` |
| **models** | 模型训练、预测、评估 | `train_eval.py`, `model_utils.py` |
| **utils** | 工具函数、日志、评估指标 | `logger.py`, `metrics.py` |
| **visualization** | 可视化分析和图表生成 | `basic_plots.py`, `error_analysis.py`, `feature_plots.py` |
| **experiments** | 实验管理和结果追踪 | `experiment.py` |

## 安装指南

### 环境要求

- **Python**: 3.8或更高版本
- **操作系统**: Windows 10+、macOS 10.14+或Ubuntu 18.04+
- **内存**: 至少4GB RAM
- **存储空间**: 预留2GB可用空间

### 依赖包

#### 核心依赖
```
pandas>=1.3.0          # 数据处理
numpy>=1.21.0           # 数值计算
scipy>=1.7.0            # 科学计算
scikit-learn>=1.0.0     # 机器学习工具
lightgbm>=3.2.0         # 梯度提升模型
torch>=1.9.0            # 深度学习框架
```

#### 可视化依赖
```
matplotlib>=3.4.0       # 基础绘图
seaborn>=0.11.0         # 统计图表
plotly>=5.0.0           # 交互式图表
```

#### 工具依赖
```
nltk>=3.6.0             # 自然语言处理
tqdm>=4.62.0            # 进度条
joblib>=1.0.0           # 并行计算
jupyter>=1.0.0          # 交互式开发
ipython>=7.25.0         # 增强交互式Python
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

程序运行完成后的输出内容：

- **predictions.csv**: 预测结果，包含每个用户对每部电影的评分预测
- **可视化图表**: 多种.png格式的分析图表
- **实验记录**: 在`experiments/`目录下保存的完整实验记录

### 可视化图表

系统自动生成的分析图表：

1. **预测效果**
   - `boxplot_true_vs_pred.png`: 真实评分与预测评分对比
   - `predicted_rating_hist.png`: 预测评分分布

2. **误差分析**
   - `prediction_error_hist.png`: 预测误差分布
   - `mean_error_per_rating.png`: 不同评分等级的平均误差
   - `confusion_heatmap.png`: 预测准确性混淆矩阵

3. **特征分析**
   - `top20_feature_importance.png`: 最重要的20个特征
   - `feature_correlation_heatmap.png`: 特征相关性热力图

## 详细使用说明

### 配置管理

系统提供灵活的配置管理，所有设置都集中在`config.py`文件中：

```python
from config import config

# 查看当前配置
print(f"模型名称: {config.model_name}")
print(f"数据路径: {config.base_dir}")
print(f"隐因子维度: {config.latent_dim}")

# 调整参数
config.n_estimators = 500      # 增加树的数量提升精度
config.learning_rate = 0.1     # 调整学习率
```

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

### 实验管理

#### 1. 创建实验

```python
from experiments.experiment import Experiment

# 创建实验
exp = Experiment("LightGBM_Baseline", config.__dict__)

# 记录指标
exp.log_metric("rmse", rmse)
exp.log_metric("mae", mae)
exp.log_metric("accuracy", accuracy)

# 保存结果
exp.save_results()
exp.save_dataframe(predictions_df, "predictions.csv")
```

#### 2. 实验比较

```python
# 加载历史实验
exp1 = Experiment.load_experiment("experiments/LightGBM_Baseline_20241201_120000")
exp2 = Experiment.load_experiment("experiments/LightGBM_Tuned_20241201_130000")

# 比较实验
comparison_fig = exp1.compare_experiments([exp2], "rmse")
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

#### 配置示例
```python
from config import Config

# 查看和修改配置
config = Config()
config.n_estimators = 500      # 调整树的数量
config.learning_rate = 0.1     # 调整学习率
config.latent_dim = 30         # 调整隐因子维度
```

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

#### 创建自定义配置类

```python
from config import Config

class CustomConfig(Config):
    def __init__(self):
        super().__init__()
        # 自定义参数
        self.n_estimators = 500
        self.learning_rate = 0.1
        self.latent_dim = 50
        
        # 自定义数据路径
        self.base_dir = "/custom/data/path"
        
        # 验证配置
        self.validate_config()

# 使用自定义配置
custom_config = CustomConfig()
```

#### 环境变量配置

```bash
# 设置环境变量
export MOVIE_DATA_DIR="/path/to/data"
export MOVIE_OUTPUT_DIR="/path/to/output"
export MOVIE_N_ESTIMATORS=500
```

```python
# 在代码中使用环境变量
import os

class EnvConfig(Config):
    def __init__(self):
        super().__init__()
        self.base_dir = os.getenv('MOVIE_DATA_DIR', self.base_dir)
        self.save_dir = os.getenv('MOVIE_OUTPUT_DIR', self.save_dir)
        self.n_estimators = int(os.getenv('MOVIE_N_ESTIMATORS', self.n_estimators))
```



## 实验管理

### 实验追踪

系统提供实验管理功能，帮助管理和比较不同的模型版本：

#### 实验文件组织
每次实验都会自动创建一个完整的文件夹，包含所有相关信息：
```
experiments/
└── LightGBM_CORAL_MovieLens_20241201_120000/
    ├── config.json          # 实验配置参数
    ├── results.json         # 性能指标结果
    ├── predictions.csv      # 详细预测结果
    ├── plots/              # 可视化图表
    │   ├── error_analysis/  #   误差分析图
    │   ├── feature_analysis/ #   特征分析图
    │   └── prediction_plots/ #   预测效果图
    ├── models/             # 训练好的模型
    │   └── lightgbm_models.pkl
    └── logs/               # 运行日志
        └── experiment.log
```

### 实验管理

项目支持实验版本管理，每次训练会自动保存：
- 实验配置 (`config.json`)
- 训练模型 (`model.pkl`) 
- 评估结果 (`results.json`)
- 预测数据 (`predictions.csv`)
- 可视化图表 (`plots/`)

```python
from experiments.experiment import Experiment

# 创建和管理实验
exp = Experiment("LightGBM_CORAL_MovieLens", config.__dict__)
exp.log_metric("rmse", rmse)
exp.save_results()
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

- **RMSE**: 0.74-0.83 (均方根误差，越小越好)
- **MAE**: 0.49-0.57 (平均绝对误差)
- **准确率**: 37%-43% (完全匹配预测)
- **训练时间**: ~3-5分钟 (标准配置)

#### 性能特点

- **高评分预测**: 4-5星评分预测相对困难，因为样本集中且差异细微
- **低评分预测**: 1-2星评分样本较少但预测相对容易
- **用户差异**: 活跃用户(评分多)的预测准确性明显高于新用户
