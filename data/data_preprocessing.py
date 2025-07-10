# data_preprocessing.py
# 数据预处理模块 - 包含异常值检测、数据清洗和质量控制功能

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
    数据预处理器类
    
    提供数据清洗、异常值检测和处理、数据质量控制等功能
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        初始化数据预处理器
        
        Args:
            contamination: 异常值比例估计（用于IsolationForest）
            random_state: 随机种子
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
        
        # 5. 多维异常检测（Isolation Forest）
        ratings_clean = self._detect_multivariate_outliers(ratings_clean)
        
        # 统计异常值信息
        self._log_outlier_statistics(ratings_clean)
        
        return ratings_clean
    
    def _basic_data_quality_check(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        基本数据质量检查
        
        检查缺失值、重复值、数据类型等基本问题
        
        Args:
            ratings: 评分数据
            
        Returns:
            ratings: 清洗后的评分数据
        """
        logger.info("执行基本数据质量检查")
        
        # 检查缺失值
        missing_counts = ratings.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"发现缺失值: {missing_counts[missing_counts > 0].to_dict()}")
            
        # 删除关键字段缺失的记录
        before_count = len(ratings)
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
        after_count = len(ratings)
        
        if before_count != after_count:
            logger.info(f"删除了 {before_count - after_count} 条关键字段缺失的记录")
        
        # 检查重复值
        duplicates = ratings.duplicated(subset=['userId', 'movieId'], keep='first')
        if duplicates.sum() > 0:
            logger.warning(f"发现 {duplicates.sum()} 条重复评分记录，保留第一条")
            ratings = ratings[~duplicates]
        
        # 标记基本质量问题
        ratings['is_basic_outlier'] = False
        
        return ratings
    
    def _detect_rating_range_outliers(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        检测评分范围异常
        
        检查评分是否在合理范围内（通常为0.5-5.0）
        
        Args:
            ratings: 评分数据
            
        Returns:
            ratings: 标记了评分范围异常的数据
        """
        logger.info("检测评分范围异常")
        
        # 检查评分范围
        valid_range = (0.5, 5.0)
        range_outliers = (ratings['rating'] < valid_range[0]) | (ratings['rating'] > valid_range[1])
        
        ratings['is_range_outlier'] = range_outliers
        
        if range_outliers.sum() > 0:
            logger.warning(f"发现 {range_outliers.sum()} 条评分范围异常记录")
            
        return ratings
    
    def _detect_user_behavior_outliers(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        检测用户行为异常
        
        分析用户评分行为模式，识别异常用户
        
        Args:
            ratings: 评分数据
            
        Returns:
            ratings: 标记了用户行为异常的数据
        """
        logger.info("检测用户行为异常")
        
        # 计算用户统计特征
        user_stats = ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'movieId': 'nunique'
        }).round(4)
        
        user_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 
                             'rating_min', 'rating_max', 'unique_movies']
        user_stats = user_stats.reset_index()
        
        # 1. 检测评分数量异常（过多或过少）
        rating_count_outliers = self._detect_outliers_zscore(
            user_stats['rating_count'], threshold=3.0
        )
        
        # 2. 检测评分方差异常（方差过小可能是机器人用户）
        rating_std_outliers = user_stats['rating_std'] < 0.1  # 标准差过小
        
        # 3. 检测评分模式异常（只给极端评分）
        extreme_rating_users = (
            (user_stats['rating_min'] >= 4.5) |  # 只给高分
            (user_stats['rating_max'] <= 2.0)    # 只给低分
        ) & (user_stats['rating_count'] >= 10)  # 且评分数量足够多
        
        # 标记异常用户
        user_stats['is_behavior_outlier'] = (
            rating_count_outliers | rating_std_outliers | extreme_rating_users
        )
        
        # 将异常标记合并回原数据
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
        
        分析评分的时间模式，识别异常时间行为
        
        Args:
            ratings: 评分数据
            
        Returns:
            ratings: 标记了时间异常的数据
        """
        logger.info("检测时间序列异常")
        
        # 确保有时间戳数据
        if 'timestamp' not in ratings.columns:
            logger.warning("缺少时间戳数据，跳过时间异常检测")
            ratings['is_temporal_outlier'] = False
            return ratings
        
        # 转换时间戳
        if 'dt' not in ratings.columns:
            ratings['dt'] = pd.to_datetime(ratings['timestamp'], unit='s')
        
        # 按用户检测评分时间间隔异常
        user_temporal_stats = []
        
        for user_id, user_ratings in ratings.groupby('userId'):
            if len(user_ratings) < 2:
                continue
                
            # 计算评分时间间隔
            user_ratings_sorted = user_ratings.sort_values('timestamp')
            time_diffs = user_ratings_sorted['timestamp'].diff().dropna()
            
            if len(time_diffs) > 0:
                # 检测异常短的时间间隔（可能是机器人行为）
                very_short_intervals = (time_diffs < 60).sum()  # 小于1分钟
                
                user_temporal_stats.append({
                    'userId': user_id,
                    'very_short_intervals': very_short_intervals,
                    'total_intervals': len(time_diffs)
                })
        
        if user_temporal_stats:
            temporal_df = pd.DataFrame(user_temporal_stats)
            temporal_df['short_interval_ratio'] = (
                temporal_df['very_short_intervals'] / temporal_df['total_intervals']
            )
            
            # 标记时间异常用户（短间隔比例过高）
            temporal_outliers = temporal_df['short_interval_ratio'] > 0.5
            temporal_df['is_temporal_outlier'] = temporal_outliers
            
            # 合并回原数据
            ratings = ratings.merge(
                temporal_df[['userId', 'is_temporal_outlier']], 
                on='userId', 
                how='left'
            )
            
            outlier_count = temporal_outliers.sum()
            if outlier_count > 0:
                logger.warning(f"发现 {outlier_count} 个时间行为异常用户")
        else:
            ratings['is_temporal_outlier'] = False
            
        return ratings
    
    def _detect_multivariate_outliers(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        使用Isolation Forest检测多维异常值
        
        Args:
            ratings: 评分数据
            
        Returns:
            ratings: 标记了多维异常的数据
        """
        logger.info("执行多维异常检测")
        
        # 准备特征用于异常检测
        user_features = ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'movieId': 'nunique'
        })
        
        user_features.columns = ['rating_count', 'rating_mean', 'rating_std', 'unique_movies']
        user_features = user_features.fillna(0).reset_index()
        
        # 标准化特征
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(
            user_features[['rating_count', 'rating_mean', 'rating_std', 'unique_movies']]
        )
        
        # 使用Isolation Forest检测异常
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        outlier_labels = iso_forest.fit_predict(features_scaled)
        user_features['is_multivariate_outlier'] = outlier_labels == -1
        
        # 合并回原数据
        ratings = ratings.merge(
            user_features[['userId', 'is_multivariate_outlier']], 
            on='userId', 
            how='left'
        )
        
        outlier_count = (outlier_labels == -1).sum()
        if outlier_count > 0:
            logger.warning(f"发现 {outlier_count} 个多维异常用户")
            
        return ratings
    
    def _detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        使用Z-score方法检测异常值
        
        Args:
            data: 数据序列
            threshold: Z-score阈值
            
        Returns:
            outlier_mask: 异常值掩码
        """
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        return z_scores > threshold
    
    def _detect_outliers_iqr(self, data: pd.Series, factor: float = 1.5) -> pd.Series:
        """
        使用IQR方法检测异常值
        
        Args:
            data: 数据序列
            factor: IQR倍数因子
            
        Returns:
            outlier_mask: 异常值掩码
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    def _log_outlier_statistics(self, ratings: pd.DataFrame) -> None:
        """
        记录异常值统计信息
        
        Args:
            ratings: 标记了异常值的评分数据
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
        处理异常值
        
        Args:
            ratings: 包含异常值标记的评分数据
            strategy: 处理策略
                - 'flag': 仅标记，不删除
                - 'remove': 删除异常值
                - 'cap': 限制异常值到合理范围
                - 'transform': 使用稳健变换
                
        Returns:
            ratings: 处理后的评分数据
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
            outlier_stats: 异常值统计报告
        """
        return self.outlier_stats


def preprocess_ratings_data(ratings: pd.DataFrame, 
                          strategy: str = 'flag',
                          contamination: float = 0.1) -> Tuple[pd.DataFrame, Dict]:
    """
    预处理评分数据的便捷函数
    
    Args:
        ratings: 原始评分数据
        strategy: 异常值处理策略
        contamination: 异常值比例估计
        
    Returns:
        processed_ratings: 处理后的评分数据
        report: 处理报告
    """
    preprocessor = DataPreprocessor(contamination=contamination)
    
    # 检测异常值
    ratings_with_outliers = preprocessor.detect_rating_outliers(ratings)
    
    # 处理异常值
    processed_ratings = preprocessor.handle_outliers(ratings_with_outliers, strategy)
    
    # 获取报告
    report = preprocessor.get_outlier_report()
    
    return processed_ratings, report