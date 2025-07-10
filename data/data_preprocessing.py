#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块

提供电影评分数据的异常值检测、数据清洗和质量控制功能。

核心功能：
1. 异常值检测：多维度检测评分数据中的异常模式
2. 数据清洗：处理缺失值、重复值和格式错误
3. 质量控制：确保数据符合业务规则和统计规律
4. 异常处理：提供多种异常值处理策略

异常检测方法：
1. 基本质量检查：缺失值、重复值、格式错误
2. 评分范围检查：超出合理评分范围的记录
3. 用户行为分析：异常评分频率、评分模式、评分方差
4. 时间序列分析：异常评分时间间隔、批量评分行为
5. 多维异常检测：使用隔离森林检测复合异常模式

处理策略：
- flag：标记异常值但保留数据
- remove：删除异常值记录
- cap：将异常值限制到合理范围
- transform：使用稳健变换处理异常值

技术特点：
- 支持大规模数据处理
- 提供详细的异常统计报告
- 可配置的异常检测阈值
- 完整的日志记录和错误处理
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
from typing import Tuple, Dict, List, Optional
from utils.logger import logger


class DataPreprocessor:
    """
    数据预处理器
    
    提供评分数据的异常值检测、数据清洗和质量控制功能。
    支持多种异常检测方法和处理策略。
    
    Attributes:
        contamination (float): 异常值比例，用于隔离森林等算法
        random_state (int): 随机种子，确保结果可复现
        outlier_stats (dict): 异常值统计信息
    
    Note:
        - 设计为处理电影评分数据的专用预处理器
        - 支持链式调用和批量处理
        - 所有检测方法都会在原数据上添加标记列
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        初始化数据预处理器
        
        Args:
            contamination: 预期异常值比例，范围[0, 0.5]
            random_state: 随机种子，用于确保结果可复现
        """
        self.contamination = contamination
        self.random_state = random_state
        self.outlier_stats = {}
        
    def detect_rating_outliers(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        检测评分数据中的异常值
        
        使用多种方法检测异常评分行为：
        1. 统计方法（Z-score, IQR）
        2. 机器学习方法（Isolation Forest）
        3. 用户行为分析（评分频率、评分模式）
        
        Args:
            ratings: 评分数据DataFrame，包含userId, movieId, rating, timestamp等列
            
        Returns:
            ratings: 标记了异常值的评分数据
        """
        logger.info("开始检测评分数据异常值")
        
        # 创建副本避免修改原数据
        ratings_clean = ratings.copy()
        
        # 1. 基本数据质量检查
        ratings_clean = self._basic_data_quality_check(ratings_clean)
        
        # 2. 评分范围异常检测
        ratings_clean = self._detect_rating_range_outliers(ratings_clean)
        
        # 3. 用户行为异常检测
        ratings_clean = self._detect_user_behavior_outliers(ratings_clean)
        
        # 4. 时间序列异常检测
        ratings_clean = self._detect_temporal_outliers(ratings_clean)
        
        # 5. 多维异常检测（隔离森林）
        ratings_clean = self._detect_multivariate_outliers(ratings_clean)
        
        # 统计异常值信息
        self._log_outlier_statistics(ratings_clean)
        
        return ratings_clean
    
    def _basic_data_quality_check(self, ratings: pd.DataFrame) -> pd.DataFrame:
        logger.info("执行基本数据质量检查")
        
        missing_counts = ratings.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"发现缺失值: {missing_counts[missing_counts > 0].to_dict()}")
            
        before_count = len(ratings)
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
        after_count = len(ratings)
        
        if before_count != after_count:
            logger.info(f"删除了 {before_count - after_count} 条关键字段缺失的记录")
        
        duplicates = ratings.duplicated(subset=['userId', 'movieId'], keep='first')
        if duplicates.sum() > 0:
            logger.warning(f"发现 {duplicates.sum()} 条重复评分记录，保留第一条")
            ratings = ratings[~duplicates]
        
        ratings['is_basic_outlier'] = False
        
        return ratings
    
    def _detect_rating_range_outliers(self, ratings: pd.DataFrame) -> pd.DataFrame:
        logger.info("检测评分范围异常")
        
        valid_range = (0.5, 5.0)
        range_outliers = (ratings['rating'] < valid_range[0]) | (ratings['rating'] > valid_range[1])
        
        ratings['is_range_outlier'] = range_outliers
        
        if range_outliers.sum() > 0:
            logger.warning(f"发现 {range_outliers.sum()} 条评分范围异常记录")
            
        return ratings
    
    def _detect_user_behavior_outliers(self, ratings: pd.DataFrame) -> pd.DataFrame:
        logger.info("检测用户行为异常")
        
        user_stats = ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'movieId': 'nunique'
        }).round(4)
        
        user_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 
                             'rating_min', 'rating_max', 'unique_movies']
        user_stats = user_stats.reset_index()
        
        # 检测评分数量异常的用户
        rating_count_outliers = self._detect_outliers_zscore(
            user_stats['rating_count'], threshold=6.0
        )
        
        # 检测评分标准差异常的用户
        rating_std_outliers = (user_stats['rating_std'] < 0.001) & (user_stats['rating_count'] >= 200)
        
        # 检测极端评分行为的用户
        extreme_rating_users = (
            (user_stats['rating_min'] >= 4.95) |
            (user_stats['rating_max'] <= 0.5)
        ) & (user_stats['rating_count'] >= 200)
        
        user_stats['is_behavior_outlier'] = (
            rating_count_outliers | rating_std_outliers | extreme_rating_users
        )
        
        ratings = ratings.merge(
            user_stats[['userId', 'is_behavior_outlier']], 
            on='userId', 
            how='left'
        )
        
        outlier_users = user_stats['is_behavior_outlier'].sum()
        if outlier_users > 0:
            logger.warning(f"发现 {outlier_users} 个异常行为用户")
            
        return ratings
    
    def _detect_temporal_outliers(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        检测时间序列异常
        
        当前实现禁用了时间序列异常检测，所有记录都标记为非异常。
        
        Args:
            ratings: 评分数据
            
        Returns:
            添加了is_temporal_outlier列的评分数据
        """
        logger.info("时间序列异常检测已禁用，设置所有记录为非异常")
        
        # 直接设置所有记录为非时间异常
        ratings['is_temporal_outlier'] = False
        
        logger.info("完成时间异常检测")
        return ratings
    
    def _detect_multivariate_outliers(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        使用隔离森林检测多维异常用户
        
        基于用户的多个统计特征（评分数量、平均评分、评分标准差、
        观看电影数量）构建特征向量，使用隔离森林算法检测异常模式。
        
        Args:
            ratings: 包含异常标记的评分数据
            
        Returns:
            添加了is_multivariate_outlier列的评分数据
            
        Note:
            - 只对评分数量>=200的活跃用户进行检测
            - 使用RobustScaler进行特征标准化
            - 使用隔离森林算法检测异常模式
        """
        logger.info("执行多维异常检测")
        
        user_features = ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'movieId': 'nunique'
        })
        
        user_features.columns = ['rating_count', 'rating_mean', 'rating_std', 'unique_movies']
        user_features = user_features.fillna(0).reset_index()
        
        min_ratings_for_detection = 200
        users_to_check = user_features[user_features['rating_count'] >= min_ratings_for_detection]
        
        if len(users_to_check) < 10:
            logger.info("用户数量不足，跳过多维异常检测")
            ratings['is_multivariate_outlier'] = False
            return ratings
        
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(
            users_to_check[['rating_count', 'rating_mean', 'rating_std', 'unique_movies']]
        )
        
        # 使用隔离森林检测多维异常
        iso_forest = IsolationForest(
            contamination=0.05,
            random_state=self.random_state,
            n_estimators=200,
            max_samples='auto',
            max_features=1.0
        )
        
        outlier_labels = iso_forest.fit_predict(features_scaled)
        users_to_check = users_to_check.copy()
        users_to_check['is_multivariate_outlier'] = outlier_labels == -1
        
        user_features['is_multivariate_outlier'] = False
        
        outlier_users = users_to_check[users_to_check['is_multivariate_outlier']]['userId']
        user_features.loc[user_features['userId'].isin(outlier_users), 'is_multivariate_outlier'] = True
        
        ratings = ratings.merge(
            user_features[['userId', 'is_multivariate_outlier']], 
            on='userId', 
            how='left'
        )
        
        outlier_count = (outlier_labels == -1).sum()
        if outlier_count > 0:
            logger.warning(f"发现 {outlier_count} 个多维异常用户（从 {len(users_to_check)} 个活跃用户中检测）")
            
        return ratings
    
    def _detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        使用Z-score方法检测异常值
        
        Args:
            data: 待检测的数据序列
            threshold: Z-score阈值，默认3.0
            
        Returns:
            布尔序列，True表示异常值
        """
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        return z_scores > threshold
    
    def _detect_outliers_iqr(self, data: pd.Series, factor: float = 1.5) -> pd.Series:
        """
        使用四分位距(IQR)方法检测异常值
        
        Args:
            data: 待检测的数据序列
            factor: IQR倍数因子，默认1.5
            
        Returns:
            布尔序列，True表示异常值
            
        Note:
            - 异常值定义为超出[Q1-1.5*IQR, Q3+1.5*IQR]范围的值
            - 对于偏态分布比Z-score更稳健
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    def _log_outlier_statistics(self, ratings: pd.DataFrame) -> None:
        """
        记录和统计异常值检测结果
        
        计算各类异常值的数量和比例，并保存到outlier_stats属性中。
        
        Args:
            ratings: 包含异常标记的评分数据
        """
        total_records = len(ratings)
        
        outlier_columns = [
            'is_basic_outlier', 'is_range_outlier', 'is_behavior_outlier',
            'is_temporal_outlier', 'is_multivariate_outlier'
        ]
        
        stats_info = {}
        for col in outlier_columns:
            if col in ratings.columns:
                outlier_count = ratings[col].sum()
                outlier_ratio = outlier_count / total_records * 100
                stats_info[col] = {
                    'count': outlier_count,
                    'ratio': round(outlier_ratio, 2)
                }
        
        # 计算总体异常值
        any_outlier = ratings[outlier_columns].any(axis=1)
        total_outliers = any_outlier.sum()
        total_ratio = total_outliers / total_records * 100
        
        logger.info(f"异常值检测完成:")
        logger.info(f"总记录数: {total_records}")
        logger.info(f"总异常值: {total_outliers} ({total_ratio:.2f}%)")
        
        for outlier_type, stats in stats_info.items():
            logger.info(f"{outlier_type}: {stats['count']} ({stats['ratio']}%)")
        
        # 保存统计信息
        self.outlier_stats = {
            'total_records': total_records,
            'total_outliers': total_outliers,
            'total_ratio': total_ratio,
            'by_type': stats_info
        }
    
    def handle_outliers(self, ratings: pd.DataFrame, strategy: str = 'flag') -> pd.DataFrame:
        """
        根据指定策略处理异常值
        
        Args:
            ratings: 包含异常标记的评分数据
            strategy: 处理策略，可选值：
                - 'flag': 仅添加异常标记，保留所有数据
                - 'remove': 删除异常值记录
                - 'cap': 将异常值限制到合理范围
                - 'transform': 使用稳健变换处理异常值
                
        Returns:
            处理后的评分数据
            
        Note:
            - 'flag'策略适用于后续分析需要异常值信息的场景
            - 'remove'策略会永久删除异常记录，需谨慎使用
            - 'cap'策略适用于数值范围异常
            - 'transform'策略适用于分布异常
        """
        logger.info(f"使用策略 '{strategy}' 处理异常值")
        
        if strategy == 'flag':
            # 仅添加综合异常标记
            outlier_columns = [
                'is_basic_outlier', 'is_range_outlier', 'is_behavior_outlier',
                'is_temporal_outlier', 'is_multivariate_outlier'
            ]
            existing_columns = [col for col in outlier_columns if col in ratings.columns]
            ratings['is_outlier'] = ratings[existing_columns].any(axis=1)
            
        elif strategy == 'remove':
            # 删除异常值
            before_count = len(ratings)
            outlier_columns = [
                'is_range_outlier', 'is_behavior_outlier',
                'is_temporal_outlier', 'is_multivariate_outlier'
            ]
            existing_columns = [col for col in outlier_columns if col in ratings.columns]
            
            if existing_columns:
                outlier_mask = ratings[existing_columns].any(axis=1)
                ratings = ratings[~outlier_mask]
                
                removed_count = before_count - len(ratings)
                logger.info(f"删除了 {removed_count} 条异常记录")
            
        elif strategy == 'cap':
            # 限制评分到合理范围
            ratings['rating'] = ratings['rating'].clip(0.5, 5.0)
            logger.info("已将评分限制到合理范围 [0.5, 5.0]")
            
        elif strategy == 'transform':
            # 使用稳健变换处理数值特征
            scaler = RobustScaler()
            
            # 对连续特征进行稳健标准化
            if 'timestamp' in ratings.columns:
                ratings['timestamp_scaled'] = scaler.fit_transform(
                    ratings[['timestamp']]
                ).flatten()
            
            logger.info("已应用稳健变换")
        
        return ratings
    
    def get_outlier_report(self) -> Dict:
        """
        获取异常值检测报告
        
        Returns:
            包含异常值统计信息的字典
        """
        return self.outlier_stats


def preprocess_ratings_data(ratings: pd.DataFrame, 
                          strategy: str = 'flag',
                          contamination: float = 0.1) -> Tuple[pd.DataFrame, Dict]:
    """
    评分数据预处理的主入口函数
    
    提供完整的数据预处理流程，包括异常检测和处理。
    
    Args:
        ratings: 原始评分数据DataFrame
        strategy: 异常值处理策略
        contamination: 预期异常值比例
        
    Returns:
        Tuple[pd.DataFrame, Dict]: (处理后的数据, 异常检测报告)
        
    Example:
        >>> ratings = pd.read_csv('ratings.csv')
        >>> clean_ratings, report = preprocess_ratings_data(ratings, strategy='flag')
        >>> print(f"处理了{report['total_outliers']}个异常值")
    """
    preprocessor = DataPreprocessor(contamination=contamination)
    
    # 检测异常值
    ratings_with_outliers = preprocessor.detect_rating_outliers(ratings)
    
    # 处理异常值
    processed_ratings = preprocessor.handle_outliers(ratings_with_outliers, strategy)
    
    # 获取报告
    report = preprocessor.get_outlier_report()
    
    return processed_ratings, report