#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电影推荐系统主程序

这是一个基于机器学习的电影评分预测系统，采用序数分类方法预测用户对电影的评分。
系统集成了多种先进的特征工程技术和可视化分析工具，提供完整的端到端解决方案。

主要功能模块：
1. 数据加载与预处理
   - 支持MovieLens数据集格式
   - 集成异常值检测和数据质量控制
   - 自动生成时间相关特征

2. 多维度特征工程
   - 协同过滤特征：基于SVD矩阵分解的用户-物品隐因子
   - 内容特征：电影类型、年份等结构化信息
   - 文本特征：基于TF-IDF的用户标签特征
   - 用户画像：用户评分行为和偏好特征
   - 电影画像：电影质量和热度特征
   - 交叉特征：用户-物品交互特征

3. 序数分类模型
   - 基于LightGBM的多二分类器架构
   - CORAL风格的序数分类方法
   - 支持类别特征和数值特征混合输入

4. 全面的评估体系
   - 多种回归和分类指标
   - 分层误差分析
   - 用户和物品维度的性能分析

5. 丰富的可视化分析
   - 预测效果可视化
   - 误差分布和模式分析
   - 特征重要性和相关性分析
   - 时间序列和热度相关性分析

6. 实验管理系统
   - 自动化实验记录和版本控制
   - 配置管理和结果追踪
   - 多实验对比分析

技术特点：
- 模块化设计，易于扩展和维护
- 完整的日志记录和错误处理
- 高效的数据处理和内存管理
- 专业的统计图表和可视化
- 支持大规模数据集处理

使用场景：
- 电影推荐系统开发
- 评分预测研究
- 推荐算法性能评估
- 特征工程实验
- 机器学习教学和研究

作者: 电影推荐系统开发团队
创建时间: 2024
最后修改: 2024
版本: 2.0
许可证: MIT
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Any
from config import config
from data.data_loader import (
    load_data, create_collaborative_filtering_features, create_content_features,
    create_tfidf_tag_features, create_user_profile_features, create_movie_profile_features,
    merge_features
)
from models.model_utils import rating_to_label, label_to_rating
from models.train_eval import train_models, predict
from utils.metrics import compute_rmse
from utils.logger import logger
from visualization.basic_plots import plot_boxplot_true_vs_pred, plot_predicted_rating_hist
from visualization.error_analysis import (
    plot_error_distribution, plot_mean_error_per_rating, plot_rmse_per_rating,
    plot_confusion_heatmap, plot_user_error_distribution, plot_error_vs_popularity,
    plot_error_by_year
)
from visualization.feature_plots import (
    plot_top20_feature_importance, plot_feature_correlation, plot_feature_distributions
)
from experiments.experiment import Experiment


def main():
    """
    主程序函数，执行完整的推荐系统训练和评估流程
    
    执行步骤：
    1. 环境初始化（目录创建、日志设置、实验记录）
    2. 数据加载和预处理（包括异常值检测）
    3. 多阶段特征工程
    4. 数据划分和模型训练
    5. 模型评估和预测
    6. 结果可视化和分析
    7. 实验结果保存
    """
    # ==================== 1. 环境初始化 ====================
    logger.info("=" * 60)
    logger.info("开始电影推荐系统训练")
    logger.info("=" * 60)
    
    # 创建输出目录
    os.makedirs(config.save_dir, exist_ok=True)
    logger.info(f"输出目录: {config.save_dir}")
    
    # 初始化实验记录
    experiment = Experiment("LightGBM_CORAL_MovieLens", config.__dict__)
    logger.info(f"实验ID: {experiment.experiment_id}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # ==================== 2. 数据加载和预处理 ====================
        logger.info("\n" + "=" * 40 + " 数据加载 " + "=" * 40)
        
        # 加载数据（包含异常值检测和处理）
        ratings, movies, tags, preprocessing_report = load_data(
            enable_preprocessing=True,  # 启用数据预处理
            outlier_strategy='flag'     # 标记异常值但不删除
        )
        
        # 记录预处理报告
        if preprocessing_report:
            experiment.log_metric("data_quality_score", preprocessing_report.get('quality_score', 0))
            experiment.log_metric("outlier_count", preprocessing_report.get('outlier_count', 0))
            experiment.log_metric("outlier_ratio", preprocessing_report.get('outlier_ratio', 0))
            logger.info(f"数据质量评分: {preprocessing_report.get('quality_score', 'N/A')}")
            logger.info(f"异常值数量: {preprocessing_report.get('outlier_count', 'N/A')}")
        
        # ==================== 3. 特征工程 ====================
        logger.info("\n" + "=" * 40 + " 特征工程 " + "=" * 40)
        
        # 3.1 协同过滤特征
        logger.info("\n--- 创建协同过滤特征 ---")
        user_f, item_f, user_bias, item_bias = create_collaborative_filtering_features(ratings)
        
        # 3.2 内容特征
        logger.info("\n--- 创建内容特征 ---")
        movies_feats, mlb = create_content_features(movies)
        
        # 3.3 TF-IDF标签特征
        logger.info("\n--- 创建TF-IDF标签特征 ---")
        rat_tag, tag_df = create_tfidf_tag_features(ratings, tags)
        
        # 3.4 用户画像特征
        logger.info("\n--- 创建用户画像特征 ---")
        user_stats, user_genre_pref = create_user_profile_features(ratings, movies)
        
        # 3.5 电影画像特征
        logger.info("\n--- 创建电影画像特征 ---")
        movie_stats = create_movie_profile_features(ratings)
        
        # 3.6 合并所有特征
        logger.info("\n--- 合并所有特征 ---")
        df = merge_features(ratings, movies_feats, user_f, item_f, user_bias, item_bias,
                          user_stats, user_genre_pref, movie_stats, tag_df, rat_tag)
    
        # ==================== 4. 数据准备和划分 ====================
        logger.info("\n" + "=" * 40 + " 数据准备 " + "=" * 40)
        
        # 4.1 添加标签（将评分转换为分类标签）
        logger.info("转换评分为分类标签...")
        df['label'] = df['rating'].apply(rating_to_label)
        logger.info(f"标签分布: {df['label'].value_counts().sort_index().to_dict()}")
        
        # 4.2 定义特征列
        logger.info("\n--- 定义特征列 ---")
        
        # 获取各类特征列
        genre_cols = mlb.classes_.tolist()
        user_genre_cols = [col for col in df.columns if col.startswith("user_genre_pref_")]
        user_f_cols = [f"user_f{i}" for i in range(config.latent_dim)]
        item_f_cols = [f"item_f{i}" for i in range(config.latent_dim)]
        cross_f_cols = [f"cross_f{i}" for i in range(config.latent_dim)]
        tag_cols = [col for col in df.columns if col.startswith("tag_")]
        
        # 组合所有特征列
        feat_cols = (
            # 基础特征
            ['year','year_r','month_r','dayofweek_r','user_bias','item_bias',
             'user_mean_rating','user_std_rating','movie_avg_rating','movie_num_ratings']
            # 各类特征
            + user_genre_cols + genre_cols + user_f_cols + item_f_cols + cross_f_cols + tag_cols
        )
        
        # 检查特征可用性
        available_features = [col for col in feat_cols if col in df.columns]
        missing_features = [col for col in feat_cols if col not in df.columns]
        
        if missing_features:
            logger.warning(f"缺失特征数量: {len(missing_features)}")
            logger.warning(f"缺失特征示例: {missing_features[:5]}")
        
        feat_cols = available_features
        logger.info(f"可用特征数量: {len(feat_cols)}")
        logger.info(f"特征类型分布:")
        logger.info(f"  - 电影类型特征: {len([c for c in feat_cols if c in genre_cols])}")
        logger.info(f"  - 用户偏好特征: {len([c for c in feat_cols if c in user_genre_cols])}")
        logger.info(f"  - 协同过滤特征: {len([c for c in feat_cols if c in user_f_cols + item_f_cols + cross_f_cols])}")
        logger.info(f"  - 标签特征: {len([c for c in feat_cols if c in tag_cols])}")
        
        # 4.3 数据划分（Leave-5-out策略）
        logger.info("\n--- 数据划分 ---")
        logger.info("使用Leave-5-out策略划分训练集和验证集...")
        
        val_indices = []
        train_indices = []
        users_with_val = 0
        users_without_val = 0
        
        for uid, group in df.groupby("userId"):
            if len(group) >= 6:  # 至少6个评分才能分出5个作为验证集
                val = group.sample(n=5, random_state=config.seed)
                train = group.drop(val.index)
                val_indices.extend(val.index)
                train_indices.extend(train.index)
                users_with_val += 1
            else:
                train_indices.extend(group.index)  # 评分少于6个的用户全部作为训练集
                users_without_val += 1
        
        logger.info(f"用户统计:")
        logger.info(f"  - 有验证集的用户: {users_with_val:,}")
        logger.info(f"  - 无验证集的用户: {users_without_val:,}")
        logger.info(f"  - 总用户数: {df['userId'].nunique():,}")
        
        # 准备训练和验证数据
        X_train = df.loc[train_indices, feat_cols].values  # 转换为numpy数组
        y_train_raw = df.loc[train_indices, 'label'].values
        X_val = df.loc[val_indices, feat_cols].values      # 转换为numpy数组
        y_val_raw = df.loc[val_indices, 'label'].values
        
        logger.info(f"数据集大小:")
        logger.info(f"  - 训练集: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"  - 验证集: {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
        
        # ==================== 5. 模型训练 ====================
        logger.info("\n" + "=" * 40 + " 模型训练 " + "=" * 40)
        
        # 训练CORAL风格的LightGBM序数分类器
        logger.info("开始训练CORAL风格的LightGBM序数分类器...")
        
        # 确定类别特征的索引（因为现在使用numpy数组）
        categorical_feature_names = ['year_r', 'month_r', 'dayofweek_r']
        categorical_indices = []
        for cat_name in categorical_feature_names:
            if cat_name in feat_cols:
                categorical_indices.append(feat_cols.index(cat_name))
        
        logger.info(f"类别特征索引: {categorical_indices}")
        
        models = train_models(
            X_train, 
            y_train_raw, 
            num_classes=config.num_classes,
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            seed=config.seed,
            categorical_features=categorical_indices if categorical_indices else None
        )
        logger.info(f"训练完成，共训练{len(models)}个二分类器")
        
        # ==================== 6. 模型预测和评估 ====================
        logger.info("\n" + "=" * 40 + " 模型评估 " + "=" * 40)
        
        # 6.1 进行预测
        logger.info("进行模型预测...")
        pred_labels = predict(models, X_val)
        pred_ratings = np.array([label_to_rating(label) for label in pred_labels])
        true_ratings = np.array([label_to_rating(label) for label in y_val_raw])
        
        # 6.2 计算评估指标
        logger.info("计算评估指标...")
        rmse = compute_rmse(true_ratings, pred_ratings)
        logger.info(f"验证集RMSE: {rmse:.4f}")
        
        # 记录详细的评估指标
        experiment.log_metric("RMSE", rmse)
        experiment.log_metric("train_size", len(X_train))
        experiment.log_metric("val_size", len(X_val))
        experiment.log_metric("feature_count", len(feat_cols))
        experiment.log_metric("users_with_val", users_with_val)
        experiment.log_metric("users_without_val", users_without_val)
        
        # 6.3 保存预测结果
        logger.info("保存预测结果...")
        output_df = pd.DataFrame({
            'userId': df.loc[val_indices, 'userId'].values,
            'movieId': df.loc[val_indices, 'movieId'].values,
            'true_rating': true_ratings,
            'pred_rating': pred_ratings,
            'error': pred_ratings - true_ratings,
            'abs_error': np.abs(pred_ratings - true_ratings)
        })
        
        # 保存到文件
        predictions_file = os.path.join(config.save_dir, "predictions.csv")
        output_df.to_csv(predictions_file, index=False)
        logger.info(f"预测结果已保存至: {predictions_file}")
        
        # 保存预测结果到实验记录
        experiment.save_dataframe(output_df, "predictions")
        
        # 输出预测结果统计
        logger.info(f"预测结果统计:")
        logger.info(f"  - 平均绝对误差: {output_df['abs_error'].mean():.4f}")
        logger.info(f"  - 误差标准差: {output_df['error'].std():.4f}")
        logger.info(f"  - 完全正确预测比例: {(output_df['abs_error'] == 0).mean()*100:.2f}%")
        logger.info(f"  - 误差≤0.5的比例: {(output_df['abs_error'] <= 0.5).mean()*100:.2f}%")

        # ==================== 7. 可视化分析 ====================
        logger.info("\n" + "=" * 40 + " 可视化分析 " + "=" * 40)
        
        try:
            # 7.1 基本预测效果图表
            logger.info("\n--- 生成基本预测效果图表 ---")
            plot_boxplot_true_vs_pred(output_df)
            plot_predicted_rating_hist(output_df)
            logger.info("基本预测效果图表生成完成")
            
            # 7.2 误差分析图表
            logger.info("\n--- 生成误差分析图表 ---")
            plot_error_distribution(output_df)
            plot_mean_error_per_rating(output_df)
            plot_rmse_per_rating(output_df)
            plot_confusion_heatmap(output_df)
            plot_user_error_distribution(output_df)
            logger.info("误差分析图表生成完成")
            
            # 7.3 特征重要性和相关性分析
            logger.info("\n--- 生成特征分析图表 ---")
            
            # 特征重要性图表
            plot_top20_feature_importance(models, X_train)
            
            # 特征相关性分析（使用前20个特征避免过多）
            correlation_features = feat_cols[:20] if len(feat_cols) > 20 else feat_cols
            plot_feature_correlation(df.loc[train_indices], correlation_features, 'rating', top_n=15)
            
            # 特征分布分析（使用前12个特征）
            distribution_features = feat_cols[:12] if len(feat_cols) > 12 else feat_cols
            plot_feature_distributions(df.loc[train_indices], distribution_features)
            
            logger.info(f"使用{len(correlation_features)}个特征进行相关性分析")
            logger.info(f"使用{len(distribution_features)}个特征进行分布分析")
            logger.info("特征分析图表生成完成")
            
            # 7.4 误差与外部因素关系分析
            logger.info("\n--- 生成误差关系分析图表 ---")
            
            # 创建电影统计数据用于热度分析
            logger.info("分析误差与电影热度的关系...")
            movie_stats_df = df.groupby('movieId').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            movie_stats_df.columns = ['movieId', 'movie_num_ratings', 'movie_avg_rating']
            plot_error_vs_popularity(output_df, movie_stats_df)
            
            # 误差与评分年份关系
            logger.info("分析误差与评分年份的关系...")
            plot_error_by_year(output_df, df, val_indices)
            
            logger.info("误差关系分析图表生成完成")
            logger.info("所有可视化图表已生成完成")
            
        except Exception as e:
            logger.error(f"可视化生成过程中出现错误: {e}")
            logger.warning("继续执行后续步骤...")
        
        # ==================== 8. 实验结果保存 ====================
        logger.info("\n" + "=" * 40 + " 实验总结 " + "=" * 40)
        
        # 8.1 记录总执行时间
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"总执行时间: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        
        # 8.2 保存完整的实验结果
        experiment_results = {
            "execution_time": total_time,
            "rmse": rmse,
            "data_stats": {
                "total_ratings": len(df),
                "total_users": df['userId'].nunique(),
                "total_movies": df['movieId'].nunique(),
                "train_size": len(X_train),
                "val_size": len(X_val)
            },
            "model_params": {
                "latent_dim": config.latent_dim,
                "tfidf_dim": config.tfidf_dim,
                "n_estimators": config.n_estimators,
                "learning_rate": config.learning_rate,
                "num_leaves": config.num_leaves,
                "num_classes": config.num_classes
            },
            "feature_stats": {
                "total_features": len(feat_cols),
                "genre_features": len([c for c in feat_cols if c in genre_cols]),
                "user_preference_features": len([c for c in feat_cols if c in user_genre_cols]),
                "collaborative_features": len([c for c in feat_cols if c in user_f_cols + item_f_cols + cross_f_cols]),
                "tag_features": len([c for c in feat_cols if c in tag_cols])
            }
        }
        
        # 如果有预处理报告，添加到实验结果中
        if preprocessing_report:
            experiment_results["preprocessing"] = preprocessing_report
        
        # 记录实验指标
        experiment.log_metric("execution_time", total_time)
        experiment.save_results(experiment_results)
        
        # 8.3 输出最终总结
        logger.info("\n" + "=" * 60)
        logger.info("实验完成！")
        logger.info("=" * 60)
        logger.info(f"最终RMSE: {rmse:.4f}")
        logger.info(f"执行时间: {total_time:.2f}秒")
        logger.info(f"使用特征数: {len(feat_cols)}")
        logger.info(f"训练样本数: {len(X_train):,}")
        logger.info(f"验证样本数: {len(X_val):,}")
        logger.info(f"结果保存目录: {config.save_dir}")
        logger.info(f"实验记录ID: {experiment.experiment_id}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        import traceback
        logger.error(f"错误详情:\n{traceback.format_exc()}")
        raise
    
    finally:
        # 确保实验记录被保存
        try:
            experiment.save_results({"status": "completed" if 'rmse' in locals() else "failed"})
        except:
            pass
    
    return rmse if 'rmse' in locals() else None


if __name__ == "__main__":
    main()
