#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电影推荐系统配置管理模块

该模块提供了电影推荐系统的全局配置管理功能，采用面向对象的设计模式，
集中管理所有系统参数，包括模型超参数、数据路径、特征工程参数、
训练配置、系统设置等。

设计特点：
- 单一配置源：所有配置参数集中管理，避免分散配置
- 类型安全：提供参数验证和类型检查
- 环境适应：自动检测硬件环境并优化配置
- 扩展性强：易于添加新的配置参数
- 文档完整：每个参数都有详细的说明和使用建议

配置分类：
1. 基础配置：模型名称、版本信息等
2. 数据路径：数据集文件路径和输出目录
3. 特征工程：各种特征提取的参数设置
4. 模型参数：LightGBM和其他算法的超参数
5. 训练配置：训练策略和优化参数
6. 系统配置：硬件设备、并行度等
7. 预处理：数据清洗和异常值处理参数
8. 可视化：图表样式和输出格式配置

使用方式：
    from config import config
    
    # 获取模型参数
    n_estimators = config.n_estimators
    learning_rate = config.learning_rate
    
    # 获取数据路径
    ratings_file = config.ratings_file
    
    # 验证配置
    config.validate_config()

注意事项：
- 修改配置后需要重新验证
- 路径配置需要确保文件存在
- 模型参数的修改会影响训练结果
- 建议在实验前备份配置

作者: 电影推荐系统开发团队
创建时间: 2024
最后修改: 2024
版本: 2.0
"""

import os
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path


class Config:
    """
    电影推荐系统配置类
    
    该类包含了整个推荐系统的所有配置参数，包括：
    - 数据文件路径配置
    - 模型超参数设置
    - 训练参数配置
    - 特征工程参数
    - 可解释性分析参数
    
    使用方式:
        config = Config()
        print(config.model_name)  # 获取模型名称
        print(config.device)      # 获取计算设备
    """

    def __init__(self):
        # ==================== 基本配置 ====================
        self.model_name = 'LightGBM_CORAL'  # 模型名称，用于实验记录和结果保存

        # ==================== 数据路径配置 ====================
        # 数据集根目录
        self.base_dir = "/Users/ming/Desktop/ml-latest-small"
        
        # 输出目录配置
        self.save_dir = "output"  # 结果保存目录
        self.save_path = os.path.join(self.save_dir, self.model_name + '.ckpt')  # 模型保存路径
        self.pred_path = os.path.join(self.save_dir, 'predictions.csv')  # 预测结果保存路径

        # 数据文件路径
        self.ratings_file = os.path.join(self.base_dir, "ratings.csv")  # 用户评分数据
        self.movies_file = os.path.join(self.base_dir, "movies.csv")   # 电影信息数据
        self.tags_file = os.path.join(self.base_dir, "tags.csv")       # 标签数据

        # ==================== 特征工程参数 ====================
        self.latent_dim = 20   # SVD协同过滤隐因子维度，控制用户和物品嵌入的维度
        self.tfidf_dim = 100   # 标签TF-IDF特征维度，控制标签特征的最大维度
        self.seed = 42         # 随机种子，确保实验可重复性
        self.num_classes = 10  # 评分类别数：半星级别(0.5, 1.0, 1.5, ..., 5.0)，用于序数分类

        # ==================== 模型训练参数 ====================
        self.n_estimators = 1000    # LightGBM树的数量，影响模型复杂度和训练时间
        self.learning_rate = 0.05   # 学习率，控制每棵树的贡献程度
        self.num_leaves = 63        # 每棵树的叶子节点数，控制树的复杂度

        # ==================== 系统配置 ====================
        # 自动检测并选择最佳计算设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==================== 可解释性分析配置 ====================
        # 构建用于可解释性图表的特征白名单
        self.whitelist_features = self.__build_whitelist()

        # ==================== 数据预处理参数 ====================
        # 异常值检测和处理相关参数
        self.outlier_detection_enabled = True    # 是否启用异常值检测
        self.outlier_handling_strategy = 'flag'  # 异常值处理策略: 'flag', 'remove', 'cap', 'transform'
        
        # 评分范围异常检测参数
        self.rating_min = 0.5    # 最小有效评分
        self.rating_max = 5.0    # 最大有效评分
        
        # 用户行为异常检测参数
        self.min_user_ratings = 1        # 用户最少评分数
        self.max_user_ratings = 10000    # 用户最多评分数（超过视为异常）
        
        # 时间异常检测参数
        self.min_timestamp = 789652009   # 最早有效时间戳 (2000年左右)
        self.max_timestamp = 2147483647  # 最晚有效时间戳 (2038年左右)

        # ==================== 初始化操作 ====================
        # 创建输出目录
        os.makedirs(self.save_dir, exist_ok=True)

    def __build_whitelist(self):
        """
        构建用于可解释性分析的特征白名单
        
        该方法定义了在生成可解释性图表（如特征重要性、相关性分析等）时
        需要重点关注的特征列表。这些特征具有明确的业务含义，便于理解和解释。
        
        Returns:
            list: 包含所有白名单特征名称的列表
        """
        # 电影类型列表 - MovieLens数据集中的标准电影类型
        genres = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        whitelist = []

        # ==================== 用户画像特征 ====================
        # 用户评分行为的统计特征
        whitelist += ['user_mean_rating', 'user_std_rating']  # 用户平均评分和评分标准差
        
        # 用户对各类型电影的偏好程度
        whitelist += [f'user_genre_pref_{g}' for g in genres]
        
        # 用户偏置项（协同过滤中的用户偏好偏移）
        whitelist += ['user_bias']

        # ==================== 电影画像特征 ====================
        # 电影质量和热度指标
        whitelist += ['movie_avg_rating', 'movie_num_ratings']  # 电影平均评分和评分数量
        
        # 电影类型特征（One-hot编码）
        whitelist += genres
        
        # 电影发行年份
        whitelist += ['year']
        
        # 电影偏置项（协同过滤中的物品偏好偏移）
        whitelist += ['item_bias']

        # ==================== 时间特征 ====================
        # 评分时间相关特征，用于捕捉时间趋势
        whitelist += ['year_r', 'month_r', 'dayofweek_r']  # 评分年份、月份、星期几

        return whitelist
    
    def get_config_dict(self):
        """
        获取配置字典，用于实验记录和结果保存
        
        Returns:
            dict: 包含所有配置参数的字典
        """
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                config_dict[attr_name] = getattr(self, attr_name)
        return config_dict
    
    def validate_config(self):
        """
        验证配置参数的有效性
        
        检查各项配置参数是否在合理范围内，确保系统能够正常运行。
        
        Raises:
            ValueError: 当配置参数无效时抛出异常
        """
        # 验证维度参数
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim必须大于0，当前值: {self.latent_dim}")
        if self.tfidf_dim <= 0:
            raise ValueError(f"tfidf_dim必须大于0，当前值: {self.tfidf_dim}")
        
        # 验证模型参数
        if self.num_classes <= 0:
            raise ValueError(f"num_classes必须大于0，当前值: {self.num_classes}")
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators必须大于0，当前值: {self.n_estimators}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate必须大于0，当前值: {self.learning_rate}")
        if self.num_leaves <= 0:
            raise ValueError(f"num_leaves必须大于0，当前值: {self.num_leaves}")
        
        # 验证评分范围
        if self.rating_min >= self.rating_max:
            raise ValueError(f"rating_min必须小于rating_max，当前值: {self.rating_min} >= {self.rating_max}")
        
        # 验证文件路径
        if not os.path.exists(self.base_dir):
            raise ValueError(f"数据目录不存在: {self.base_dir}")
        
        return True

# 实例化全局配置对象
config = Config()
