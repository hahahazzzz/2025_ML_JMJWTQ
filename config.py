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
    电影推荐系统配置管理类
    
    包含数据路径、模型参数、特征工程配置等所有系统配置项。
    支持配置验证和白名单管理。
    """

    def __init__(self):
        # ==================== 模型配置 ====================
        self.model_name = 'LightGBM_CORAL'  # 模型名称，用于保存和识别模型

        # ==================== 路径配置 ====================
        self.base_dir = "data"  # 数据文件根目录
        self.save_dir = "output"  # 输出文件保存目录
        self.save_path = os.path.join(self.save_dir, self.model_name + '.ckpt')  # 模型保存路径
        self.pred_path = os.path.join(self.save_dir, 'predictions.csv')  # 预测结果保存路径

        # 数据文件路径配置
        self.ratings_file = os.path.join(self.base_dir, "ratings.csv")  # 用户评分数据文件
        self.movies_file = os.path.join(self.base_dir, "movies.csv")  # 电影信息数据文件
        self.tags_file = os.path.join(self.base_dir, "tags.csv")  # 用户标签数据文件

        # ==================== 特征工程配置 ====================
        self.latent_dim = 20  # 协同过滤潜在因子维度，控制用户和物品嵌入向量的大小
        self.tfidf_dim = 100  # TF-IDF特征维度，用于标签文本特征提取
        self.seed = 42  # 随机种子，确保实验结果可复现
        self.num_classes = 10  # 分类数量，对应评分0.5-5.0的10个等级

        # ==================== LightGBM模型参数 ====================
        self.n_estimators = 1000  # 树的数量，更多的树通常能提高精度但增加训练时间
        self.learning_rate = 0.05  # 学习率，控制每棵树的贡献程度，较小值需要更多树
        self.num_leaves = 63  # 每棵树的叶子节点数，控制模型复杂度，建议小于2^max_depth

        # ==================== 硬件配置 ====================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 计算设备选择
        
        # ==================== 特征白名单配置 ====================
        self.whitelist_features = self.__build_whitelist()  # 可解释特征白名单，用于特征重要性分析

        # ==================== 数据预处理配置 ====================
        self.outlier_detection_enabled = True  # 是否启用异常值检测
        self.outlier_handling_strategy = 'flag'  # 异常值处理策略：'flag'标记，'remove'移除，'clip'截断
        
        # 评分范围约束
        self.rating_min = 0.5  # 最小评分值
        self.rating_max = 5.0  # 最大评分值
        
        # 用户评分数量约束
        self.min_user_ratings = 1  # 用户最少评分数量
        self.max_user_ratings = 10000  # 用户最多评分数量（用于异常检测）
        
        # 时间戳范围约束（Unix时间戳）
        self.min_timestamp = 789652009  # 最小时间戳（约2005年）
        self.max_timestamp = 2147483647  # 最大时间戳（约2038年）

        # 确保输出目录存在
        os.makedirs(self.save_dir, exist_ok=True)

    def __build_whitelist(self):
        """
        构建可解释特征白名单
        
        该方法创建一个包含所有可解释特征名称的列表，这些特征在特征重要性
        分析中会被单独显示，而不是被归类为隐因子特征。
        
        Returns:
            List[str]: 可解释特征名称列表
        """
        # 电影类型列表（基于MovieLens数据集的标准类型）
        genres = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        whitelist = []
        
        # 用户统计特征
        whitelist += ['user_mean_rating', 'user_std_rating']  # 用户平均评分和评分标准差
        
        # 用户类型偏好特征
        whitelist += [f'user_genre_pref_{g}' for g in genres]  # 用户对各类型电影的偏好程度
        
        # 偏置特征
        whitelist += ['user_bias']  # 用户偏置（用户整体评分倾向）
        
        # 电影统计特征
        whitelist += ['movie_avg_rating', 'movie_num_ratings']  # 电影平均评分和评分数量
        
        # 电影类型特征
        whitelist += genres  # 电影所属类型的one-hot编码
        
        # 时间特征
        whitelist += ['year']  # 电影发行年份
        
        # 物品偏置特征
        whitelist += ['item_bias']  # 物品偏置（电影整体受欢迎程度）
        
        # 评分时间特征
        whitelist += ['year_r', 'month_r', 'dayofweek_r']  # 评分年份、月份、星期几

        return whitelist
    
    def get_config_dict(self):
        """
        获取配置字典
        
        将配置对象的所有非私有、非方法属性转换为字典格式，
        便于序列化、日志记录和配置比较。
        
        Returns:
            Dict[str, Any]: 包含所有配置参数的字典
        """
        config_dict = {}
        for attr_name in dir(self):
            # 跳过私有属性（以_开头）和方法
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                config_dict[attr_name] = getattr(self, attr_name)
        return config_dict
    
    def validate_config(self):
        """
        验证配置参数的有效性
        
        检查所有配置参数是否在合理范围内，确保系统能够正常运行。
        在系统启动时调用此方法可以提前发现配置错误。
        
        Returns:
            bool: 验证通过返回True
            
        Raises:
            ValueError: 当配置参数无效时抛出异常
        """
        # 验证特征工程参数
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim必须大于0，当前值: {self.latent_dim}")
        if self.tfidf_dim <= 0:
            raise ValueError(f"tfidf_dim必须大于0，当前值: {self.tfidf_dim}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes必须大于0，当前值: {self.num_classes}")
            
        # 验证模型参数
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators必须大于0，当前值: {self.n_estimators}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate必须大于0，当前值: {self.learning_rate}")
        if self.num_leaves <= 0:
            raise ValueError(f"num_leaves必须大于0，当前值: {self.num_leaves}")
            
        # 验证数据范围参数
        if self.rating_min >= self.rating_max:
            raise ValueError(f"rating_min必须小于rating_max，当前值: {self.rating_min} >= {self.rating_max}")
            
        # 验证路径存在性
        if not os.path.exists(self.base_dir):
            raise ValueError(f"数据目录不存在: {self.base_dir}")
            
        return True

# 实例化全局配置对象
config = Config()
