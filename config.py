#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电影推荐系统配置管理模块

提供全局配置管理功能，集中管理系统参数
"""

import os
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path


class Config:
    """
    电影推荐系统配置类
    
    包含系统的所有配置参数
    """

    def __init__(self):
        # ==================== 基本配置 ====================
        self.model_name = 'LightGBM_CORAL'  # 模型名称，用于实验记录和结果保存

        # ==================== 数据路径配置 ====================
        # 数据集根目录
        self.base_dir = "data"
        
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
        
        Returns:
            包含白名单特征名称的列表
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
        获取配置字典
        
        Returns:
            包含所有配置参数的字典
        """
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                config_dict[attr_name] = getattr(self, attr_name)
        return config_dict
    
    def validate_config(self):
        """
        验证配置参数的有效性
        
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
