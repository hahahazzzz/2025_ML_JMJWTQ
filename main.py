#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电影推荐系统主程序

基于机器学习的电影评分预测系统

系统架构：
1. 数据加载与预处理：加载MovieLens数据集，进行数据清洗和异常值检测
2. 特征工程：构建多维度特征体系
   - 协同过滤特征：SVD分解得到用户和物品隐因子
   - 内容特征：电影年份、类型独热编码
   - 文本特征：用户标签的TF-IDF向量化
   - 统计特征：用户和电影的评分统计信息
   - 偏好特征：用户对不同类型电影的偏好
   - 时间特征：评分时间的年、月、星期信息
3. 模型训练：使用CORAL序数分类方法，训练多个LightGBM二分类器
4. 模型评估：使用Leave-5-out验证策略，计算RMSE等指标
5. 结果可视化：生成预测效果、误差分析、特征重要性等图表

技术特点：
- 序数分类：将评分预测建模为序数分类问题，保持评分间的顺序关系
- 多维特征：融合协同过滤、内容、文本、统计等多种特征
- 异常值处理：使用Isolation Forest检测和标记异常评分
- 中文可视化：支持中文字体的图表生成

输出文件：
- predictions.csv：详细的预测结果
- 多个PNG图表文件：可视化分析结果
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
from visualization.font_config import setup_chinese_fonts
from visualization.basic_plots import plot_boxplot_true_vs_pred, plot_predicted_rating_hist
from visualization.error_analysis import (
    plot_error_distribution, plot_mean_error_per_rating, plot_rmse_per_rating,
    plot_confusion_heatmap, plot_user_error_distribution, plot_error_vs_popularity
)
from visualization.feature_plots import (
    plot_top20_feature_importance, plot_feature_correlation, plot_feature_distributions
)
# 注意：实验管理功能已移除，结果直接保存到output目录


def main():
    """
    电影推荐系统主程序
    
    执行完整的机器学习流水线，从数据加载到结果可视化。
    
    主要步骤：
    1. 环境初始化：创建输出目录，设置日志
    2. 数据加载：读取评分、电影、标签数据，进行预处理
    3. 特征工程：生成多维度特征矩阵
    4. 数据划分：使用Leave-5-out策略划分训练/验证集
    5. 模型训练：训练CORAL风格的LightGBM序数分类器
    6. 模型评估：计算RMSE等评估指标
    7. 结果保存：保存预测结果到CSV文件
    8. 可视化分析：生成多种分析图表
    9. 实验总结：记录完整的实验结果和统计信息
    
    Returns:
        float: 验证集RMSE，如果执行失败则返回None
    
    Raises:
        Exception: 数据加载、特征工程、模型训练等步骤中的任何错误
    
    Note:
        - 使用Leave-5-out验证：对评分>=6的用户随机选5个评分作验证
        - CORAL序数分类：训练多个二分类器预测评分是否>=某阈值
        - 特征白名单：只使用配置中指定的特征子集
        - 异常值处理：可选的异常值检测和标记功能
    """
    # ========== 1. 环境初始化 ==========
    logger.info("开始电影推荐系统训练")
    logger.info(f"输出目录: {config.save_dir}")
    logger.info(f"随机种子: {config.seed}")
    os.makedirs(config.save_dir, exist_ok=True)
    start_time = time.time()
    
    try:
        # ========== 2. 数据加载和预处理 ==========
        logger.info("数据加载和预处理")
        logger.info(f"数据文件路径:")
        logger.info(f"  - 评分数据: {config.ratings_file}")
        logger.info(f"  - 电影数据: {config.movies_file}")
        logger.info(f"  - 标签数据: {config.tags_file}")
        ratings, movies, tags, preprocessing_report = load_data(
            enable_preprocessing=True,
            outlier_strategy='flag'
        )
        
        if preprocessing_report:
            logger.info(f"数据质量评分: {preprocessing_report.get('quality_score', 0)}")
            logger.info(f"异常值数量: {preprocessing_report.get('outlier_count', 0)}")
            logger.info(f"异常值比例: {preprocessing_report.get('outlier_ratio', 0)}")
        
        # ========== 3. 特征工程 - 创建多维度特征体系 ==========
        logger.info("特征工程")
        logger.info("开始构建多维度特征体系...")
        
        # 3.1 协同过滤特征：使用SVD分解生成用户和物品隐因子
        logger.info("3.1 生成协同过滤特征（SVD分解）")
        user_f, item_f, user_bias, item_bias = create_collaborative_filtering_features(ratings)
        
        # 3.2 内容特征：电影年份和类型独热编码
        logger.info("3.2 生成内容特征（年份+类型）")
        movies_feats, mlb = create_content_features(movies)
        
        # 3.3 文本特征：用户标签的TF-IDF向量化
        logger.info("3.3 生成文本特征（TF-IDF标签）")
        rat_tag, tag_df = create_tfidf_tag_features(ratings, tags)
        
        # 3.4 用户画像特征：评分统计和类型偏好
        logger.info("3.4 生成用户画像特征")
        user_stats, user_genre_pref = create_user_profile_features(ratings, movies)
        
        # 3.5 电影画像特征：评分统计信息
        logger.info("3.5 生成电影画像特征")
        movie_stats = create_movie_profile_features(ratings)
        
        # 3.6 特征合并：整合所有特征并生成交叉特征
        logger.info("3.6 合并所有特征并生成交叉特征")
        df = merge_features(ratings, movies_feats, user_f, item_f, user_bias, item_bias,
                          user_stats, user_genre_pref, movie_stats, tag_df, rat_tag)
    
        # ========== 4. 数据准备和划分 ==========
        logger.info("数据准备和划分")
        
        # 4.1 评分标签转换：将连续评分转换为序数分类标签
        # 评分映射：0.5->0, 1.0->1, 1.5->2, ..., 5.0->9
        df['label'] = df['rating'].apply(rating_to_label)
        logger.info(f"标签分布: {df['label'].value_counts().sort_index().to_dict()}")
        
        # 4.2 特征列表构建：整理所有可用特征
        genre_cols = mlb.classes_.tolist()  # 电影类型特征
        user_genre_cols = [col for col in df.columns if col.startswith("user_genre_pref_")]  # 用户类型偏好
        user_f_cols = [f"user_f{i}" for i in range(config.latent_dim)]  # 用户潜在因子
        item_f_cols = [f"item_f{i}" for i in range(config.latent_dim)]  # 物品潜在因子
        cross_f_cols = [f"cross_f{i}" for i in range(config.latent_dim)]  # 交叉特征
        tag_cols = [col for col in df.columns if col.startswith("tag_")]  # TF-IDF标签特征
        feat_cols = (
            ['year','year_r','month_r','dayofweek_r','user_bias','item_bias',
             'user_mean_rating','user_std_rating','movie_avg_rating','movie_num_ratings']
            + user_genre_cols + genre_cols + user_f_cols + item_f_cols + cross_f_cols + tag_cols
        )
        
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
        
        # 4.3 数据集划分：使用Leave-5-out策略
        # 策略说明：对于评分数>=6的用户，随机选择5个评分作为验证集，其余作为训练集
        # 这样可以确保验证集中的用户在训练集中也有足够的历史数据
        logger.info("使用Leave-5-out策略划分训练集和验证集")
        
        val_indices = []
        train_indices = []
        users_with_val = 0
        users_without_val = 0
        
        for uid, group in df.groupby("userId"):
            if len(group) >= 6:  # 用户评分数足够多，可以划分验证集
                val = group.sample(n=5, random_state=config.seed)
                train = group.drop(val.index)
                val_indices.extend(val.index)
                train_indices.extend(train.index)
                users_with_val += 1
            else:  # 用户评分数较少，全部用于训练
                train_indices.extend(group.index)
                users_without_val += 1
        
        logger.info(f"用户统计:")
        logger.info(f"  - 有验证集的用户: {users_with_val:,}")
        logger.info(f"  - 无验证集的用户: {users_without_val:,}")
        logger.info(f"  - 总用户数: {df['userId'].nunique():,}")
        
        X_train_df = df.loc[train_indices, feat_cols]
        X_train = X_train_df.values
        y_train_raw = df.loc[train_indices, 'label'].values
        X_val = df.loc[val_indices, feat_cols].values
        y_val_raw = df.loc[val_indices, 'label'].values
        
        logger.info(f"数据集大小:")
        logger.info(f"  - 训练集: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"  - 验证集: {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
        
        # ========== 5. 模型训练 - CORAL序数分类 ==========
        # CORAL方法：训练多个二分类器，每个分类器预测评分是否>=某个阈值
        # 优势：保持评分间的顺序关系，比多分类更适合序数数据
        logger.info("模型训练")
        logger.info("开始训练CORAL风格的LightGBM序数分类器")
        logger.info(f"将训练{config.num_classes-1}个二分类器")
        
        # 5.1 类别特征指定：告知LightGBM哪些特征是类别型
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
        
        # ========== 6. 模型评估 ==========
        logger.info("模型评估")
        logger.info("进行模型预测")
        pred_labels = predict(models, X_val)
        pred_ratings = np.array([label_to_rating(label) for label in pred_labels])
        true_ratings = np.array([label_to_rating(label) for label in y_val_raw])
        
        logger.info("计算评估指标")
        rmse = compute_rmse(true_ratings, pred_ratings)
        logger.info(f"验证集RMSE: {rmse:.4f}")
        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"验证集大小: {len(X_val)}")
        logger.info(f"特征数量: {len(feat_cols)}")
        logger.info(f"有验证集的用户数: {users_with_val}")
        logger.info(f"无验证集的用户数: {users_without_val}")
        
        # ========== 7. 结果保存 ==========
        logger.info("保存预测结果")
        output_df = pd.DataFrame({
            'userId': df.loc[val_indices, 'userId'].values,
            'movieId': df.loc[val_indices, 'movieId'].values,
            'true_rating': true_ratings,
            'pred_rating': pred_ratings,
            'error': pred_ratings - true_ratings,
            'abs_error': np.abs(pred_ratings - true_ratings)
        })
        
        predictions_file = os.path.join(config.save_dir, "predictions.csv")
        output_df.to_csv(predictions_file, index=False)
        logger.info(f"预测结果已保存至: {predictions_file}")
        
        # 预测结果已保存到CSV文件
        
        logger.info(f"预测结果统计:")
        logger.info(f"  - 平均绝对误差: {output_df['abs_error'].mean():.4f}")
        logger.info(f"  - 误差标准差: {output_df['error'].std():.4f}")
        logger.info(f"  - 完全正确预测比例: {(output_df['abs_error'] == 0).mean()*100:.2f}%")
        logger.info(f"  - 误差≤0.5的比例: {(output_df['abs_error'] <= 0.5).mean()*100:.2f}%")

        # ========== 8. 可视化分析 ==========
        logger.info("可视化分析")
        logger.info("设置中文字体配置")
        setup_chinese_fonts()
        
        try:
            logger.info("生成基本预测效果图表")
            plot_boxplot_true_vs_pred(output_df)
            plot_predicted_rating_hist(output_df)
            logger.info("基本预测效果图表生成完成")
            
            logger.info("生成误差分析图表")
            plot_error_distribution(output_df)
            plot_mean_error_per_rating(output_df)
            plot_rmse_per_rating(output_df)
            plot_confusion_heatmap(output_df)
            plot_user_error_distribution(output_df)
            logger.info("误差分析图表生成完成")
            
            logger.info("生成特征分析图表")
            plot_top20_feature_importance(models, X_train_df)
            
            correlation_features = feat_cols[:20] if len(feat_cols) > 20 else feat_cols
            plot_feature_correlation(X_train_df[correlation_features], y_train_raw, whitelist_features=correlation_features)
            
            distribution_features = feat_cols[:12] if len(feat_cols) > 12 else feat_cols
            plot_feature_distributions(X_train_df[distribution_features], whitelist_features=distribution_features)
            
            logger.info(f"使用{len(correlation_features)}个特征进行相关性分析")
            logger.info(f"使用{len(distribution_features)}个特征进行分布分析")
            logger.info("特征分析图表生成完成")
            
            logger.info("生成误差关系分析图表")
            logger.info("分析误差与电影热度的关系")
            movie_stats_df = df.groupby('movieId').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            movie_stats_df.columns = ['movieId', 'movie_num_ratings', 'movie_avg_rating']
            plot_error_vs_popularity(output_df, movie_stats_df)
            

            
            logger.info("误差关系分析图表生成完成")
            logger.info("所有可视化图表已生成完成")
            
        except Exception as e:
            logger.error(f"可视化生成过程中出现错误: {e}")
            logger.warning("继续执行后续步骤...")
        
        # ========== 9. 实验总结 ==========
        logger.info("实验总结")
        
        # 记录总执行时间
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"总执行时间: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        
        # 保存完整的实验结果
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
        
        # 输出最终总结
        logger.info("实验完成！")
        logger.info(f"最终RMSE: {rmse:.4f}")
        logger.info(f"执行时间: {total_time:.2f}秒")
        logger.info(f"使用特征数: {len(feat_cols)}")
        logger.info(f"训练样本数: {len(X_train):,}")
        logger.info(f"验证样本数: {len(X_val):,}")
        logger.info(f"结果保存目录: {config.save_dir}")
        logger.info("实验总结完成")
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        import traceback
        logger.error(f"错误详情:\n{traceback.format_exc()}")
        raise
    
    finally:
        pass
    
    return rmse if 'rmse' in locals() else None


if __name__ == "__main__":
    main()
