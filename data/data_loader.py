# data_loader.py
# 数据加载和特征工程模块
# 
# 本模块负责:
# 1. 从CSV文件加载原始数据（评分、电影、标签）
# 2. 创建多种类型的特征：协同过滤、内容特征、TF-IDF、用户画像、电影画像
# 3. 特征合并和交叉特征生成
# 4. 集成数据预处理和异常值处理功能

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional
from config import config
from .data_preprocessing import preprocess_ratings_data
from utils.logger import logger


def load_data(enable_preprocessing: bool = True, 
              outlier_strategy: str = 'flag') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """
    加载评分、电影、标签数据，并进行数据预处理
    
    该函数执行以下步骤：
    1. 从CSV文件加载原始数据
    2. 添加时间相关特征（年份、月份、星期几）
    3. 可选的数据预处理和异常值检测
    4. 数据质量验证
    
    Args:
        enable_preprocessing: 是否启用数据预处理和异常值检测
        outlier_strategy: 异常值处理策略
            - 'flag': 仅标记异常值，不删除
            - 'remove': 删除检测到的异常值
            - 'cap': 将异常值限制到合理范围
            - 'transform': 使用稳健变换处理异常值
    
    Returns:
        ratings: 评分数据DataFrame，包含时间特征和可选的异常值标记
        movies: 电影数据DataFrame，包含电影ID、标题、类型信息
        tags: 标签数据DataFrame，包含用户标签信息
        preprocessing_report: 数据预处理报告（如果启用预处理）
    
    Raises:
        FileNotFoundError: 当数据文件不存在时
        ValueError: 当数据格式不正确时
    """
    logger.info("开始加载数据文件")
    
    try:
        # 1. 加载评分数据
        logger.info(f"加载评分数据: {config.ratings_file}")
        ratings = pd.read_csv(config.ratings_file)
        
        # 验证评分数据必要列
        required_rating_cols = ['userId', 'movieId', 'rating', 'timestamp']
        missing_cols = [col for col in required_rating_cols if col not in ratings.columns]
        if missing_cols:
            raise ValueError(f"评分数据缺少必要列: {missing_cols}")
        
        # 2. 添加时间相关特征
        logger.info("生成时间相关特征")
        ratings['dt'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['year_r'] = ratings['dt'].dt.year  # 评分年份
        ratings['month_r'] = ratings['dt'].dt.month  # 评分月份
        ratings['dayofweek_r'] = ratings['dt'].dt.dayofweek  # 评分星期几（0=周一）
        
        # 3. 加载电影数据
        logger.info(f"加载电影数据: {config.movies_file}")
        movies = pd.read_csv(config.movies_file)
        
        # 验证电影数据必要列
        required_movie_cols = ['movieId', 'title', 'genres']
        missing_cols = [col for col in required_movie_cols if col not in movies.columns]
        if missing_cols:
            raise ValueError(f"电影数据缺少必要列: {missing_cols}")
        
        # 4. 加载标签数据
        logger.info(f"加载标签数据: {config.tags_file}")
        tags = pd.read_csv(config.tags_file)
        
        # 验证标签数据必要列
        required_tag_cols = ['userId', 'movieId', 'tag']
        missing_cols = [col for col in required_tag_cols if col not in tags.columns]
        if missing_cols:
            logger.warning(f"标签数据缺少列: {missing_cols}，可能影响TF-IDF特征生成")
        
        # 5. 数据预处理（可选）
        preprocessing_report = None
        if enable_preprocessing:
            logger.info("开始数据预处理和异常值检测")
            ratings, preprocessing_report = preprocess_ratings_data(
                ratings, 
                strategy=outlier_strategy,
                contamination=0.1
            )
            logger.info("数据预处理完成")
        
        # 6. 记录数据统计信息
        logger.info(f"数据加载完成:")
        logger.info(f"  评分记录数: {len(ratings):,}")
        logger.info(f"  电影数量: {len(movies):,}")
        logger.info(f"  标签记录数: {len(tags):,}")
        logger.info(f"  用户数量: {ratings['userId'].nunique():,}")
        logger.info(f"  评分范围: {ratings['rating'].min():.1f} - {ratings['rating'].max():.1f}")
        logger.info(f"  时间范围: {ratings['dt'].min()} - {ratings['dt'].max()}")
        
        return ratings, movies, tags, preprocessing_report
        
    except FileNotFoundError as e:
        logger.error(f"数据文件未找到: {e}")
        raise
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise


def create_collaborative_filtering_features(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    使用SVD（奇异值分解）创建协同过滤特征
    
    该函数通过矩阵分解技术将用户-物品评分矩阵分解为低维的用户和物品隐因子，
    同时提取用户和物品的偏置项，这些特征能够捕捉用户偏好和物品特性的潜在模式。
    
    算法流程：
    1. 将评分数据转换为Surprise库格式
    2. 使用SVD算法进行矩阵分解，学习用户和物品的隐因子表示
    3. 提取用户隐因子、物品隐因子、用户偏置和物品偏置
    4. 将内部ID映射回原始ID，并转换为DataFrame格式
    
    Args:
        ratings: 评分数据DataFrame，必须包含['userId', 'movieId', 'rating']列
    
    Returns:
        user_f: 用户隐因子特征DataFrame，包含userId和user_f0到user_f{latent_dim-1}列
               每行代表一个用户在隐空间中的表示向量
        item_f: 电影隐因子特征DataFrame，包含movieId和item_f0到item_f{latent_dim-1}列
               每行代表一个电影在隐空间中的表示向量
        user_bias: 用户偏置DataFrame，包含userId和user_bias列
                  反映用户的整体评分倾向（严格或宽松）
        item_bias: 电影偏置DataFrame，包含movieId和item_bias列
                  反映电影的整体质量水平（高分或低分倾向）
    
    Note:
        - 隐因子维度由config.latent_dim控制
        - 评分范围设置为0.5-5.0以匹配MovieLens数据集
        - SVD能够处理稀疏评分矩阵并发现潜在的用户-物品关联模式
        - 偏置项有助于捕捉用户和物品的全局特性
    """
    logger.info(f"开始创建协同过滤特征，隐因子维度: {config.latent_dim}")
    
    try:
        # 1. 准备Surprise数据格式
        # Reader定义评分范围，确保数据正确解析
        reader = Reader(rating_scale=(0.5, 5.0))
        data_surp = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        trainset = data_surp.build_full_trainset()
        
        logger.info(f"训练集统计: {trainset.n_users}个用户, {trainset.n_items}个物品, {trainset.n_ratings}条评分")
        
        # 2. 训练SVD模型
        # SVD通过梯度下降优化用户和物品隐因子以及偏置项
        svd = SVD(n_factors=config.latent_dim, random_state=config.seed)
        
        logger.info("开始训练SVD模型...")
        svd.fit(trainset)
        logger.info("SVD模型训练完成")

        # 3. 创建ID映射表
        # Surprise内部使用连续整数ID，需要映射回原始用户和电影ID
        u_map = {i: trainset.to_raw_uid(i) for i in trainset.all_users()}
        i_map = {i: trainset.to_raw_iid(i) for i in trainset.all_items()}
        
        logger.info("提取用户和物品隐因子...")
        
        # 4. 提取用户隐因子特征
        # svd.pu: 用户隐因子矩阵，形状为(n_users, n_factors)
        user_f = pd.DataFrame(
            svd.pu, 
            index=[u_map[i] for i in range(len(svd.pu))],
            columns=[f"user_f{i}" for i in range(config.latent_dim)]
        ).reset_index().rename(columns={'index': 'userId'})
        
        # 5. 提取物品隐因子特征
        # svd.qi: 物品隐因子矩阵，形状为(n_items, n_factors)
        item_f = pd.DataFrame(
            svd.qi, 
            index=[i_map[i] for i in range(len(svd.qi))],
            columns=[f"item_f{i}" for i in range(config.latent_dim)]
        ).reset_index().rename(columns={'index': 'movieId'})

        # 6. 提取用户偏置
        # svd.bu: 用户偏置向量，反映用户的评分倾向
        user_bias = pd.DataFrame({
            'userId': [u_map[i] for i in range(len(svd.bu))], 
            'user_bias': svd.bu
        })
        
        # 7. 提取物品偏置
        # svd.bi: 物品偏置向量，反映物品的质量水平
        item_bias = pd.DataFrame({
            'movieId': [i_map[i] for i in range(len(svd.bi))], 
            'item_bias': svd.bi
        })
        
        logger.info(f"协同过滤特征创建完成:")
        logger.info(f"  用户隐因子: {user_f.shape}")
        logger.info(f"  物品隐因子: {item_f.shape}")
        logger.info(f"  用户偏置: {user_bias.shape}")
        logger.info(f"  物品偏置: {item_bias.shape}")
        
        return user_f, item_f, user_bias, item_bias
        
    except Exception as e:
        logger.error(f"协同过滤特征创建失败: {e}")
        raise


def create_content_features(movies: pd.DataFrame) -> Tuple[pd.DataFrame, MultiLabelBinarizer]:
    """
    创建电影内容特征，包括年份和类型特征
    
    该函数从电影标题中提取年份信息，并将电影类型转换为One-Hot编码特征。
    这些内容特征能够帮助模型理解电影的基本属性和类型偏好。
    
    处理流程：
    1. 从电影标题中提取发行年份
    2. 对年份进行标准化处理
    3. 将电影类型字符串分割并进行One-Hot编码
    4. 合并所有内容特征
    
    Args:
        movies: 电影数据DataFrame，必须包含['movieId', 'title', 'genres']列
               title格式应为"Movie Title (Year)"
               genres格式应为"Genre1|Genre2|Genre3"
    
    Returns:
        movies_feats: 内容特征DataFrame，包含：
                     - movieId: 电影ID
                     - year: 标准化后的发行年份
                     - 各类型的One-Hot编码特征
        mlb: 多标签二值化器，用于类型编码的转换器对象
    
    Note:
        - 缺失年份使用均值填充
        - 年份标准化有助于模型训练稳定性
        - One-Hot编码能够捕捉不同类型的偏好模式
        - 返回的mlb可用于新数据的类型编码
    """
    logger.info("开始创建电影内容特征")
    
    try:
        # 创建电影数据副本，避免修改原始数据
        movies_copy = movies.copy()
        
        # 1. 提取年份信息
        logger.info("提取电影发行年份...")
        # 使用正则表达式从标题中提取年份，格式如"Movie Title (1995)"
        movies_copy['year'] = movies_copy['title'].str.extract(r'\((\d{4})\)', expand=False).astype(float)
        
        # 统计年份提取情况
        missing_years = movies_copy['year'].isna().sum()
        if missing_years > 0:
            logger.warning(f"有{missing_years}部电影缺少年份信息，将使用均值填充")
        
        # 2. 标准化年份
        logger.info("标准化年份特征...")
        scaler = StandardScaler()
        # 使用均值填充缺失年份，然后进行标准化
        year_filled = movies_copy[['year']].fillna(movies_copy['year'].mean())
        movies_copy['year'] = scaler.fit_transform(year_filled).flatten()
        
        # 3. 处理电影类型
        logger.info("处理电影类型特征...")
        # 将类型字符串按'|'分割为列表
        movies_copy['genres'] = movies_copy['genres'].str.split('|')
        
        # 使用MultiLabelBinarizer进行One-Hot编码
        mlb = MultiLabelBinarizer()
        genre_features = mlb.fit_transform(movies_copy['genres'])
        
        # 创建类型特征DataFrame
        genres_ohe = pd.DataFrame(
            genre_features,
            columns=mlb.classes_, 
            index=movies_copy.index
        )
        
        logger.info(f"识别到{len(mlb.classes_)}种电影类型: {list(mlb.classes_)}")
        
        # 4. 合并所有内容特征
        logger.info("合并内容特征...")
        movies_feats = pd.concat([movies_copy[['movieId', 'year']], genres_ohe], axis=1)
        
        logger.info(f"内容特征创建完成:")
        logger.info(f"  特征维度: {movies_feats.shape}")
        logger.info(f"  年份范围: {movies_copy['year'].min():.2f} - {movies_copy['year'].max():.2f} (标准化后)")
        logger.info(f"  类型数量: {len(mlb.classes_)}")
        
        return movies_feats, mlb
        
    except Exception as e:
        logger.error(f"内容特征创建失败: {e}")
        raise


def create_tfidf_tag_features(ratings: pd.DataFrame, tags: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    创建基于TF-IDF的标签特征
    
    该函数将用户为电影添加的文本标签转换为数值特征，通过TF-IDF算法
    捕捉标签的重要性和区分度，为推荐系统提供丰富的内容语义信息。
    
    处理流程：
    1. 按用户-电影对聚合标签，形成每个评分记录的标签文档
    2. 将标签文档与评分数据合并
    3. 使用TF-IDF向量化器处理标签文本
    4. 生成固定维度的TF-IDF特征矩阵
    
    Args:
        ratings: 评分数据DataFrame，必须包含['userId', 'movieId']列
        tags: 标签数据DataFrame，必须包含['userId', 'movieId', 'tag']列
              每行代表一个用户为某部电影添加的标签
    
    Returns:
        rat_tag: 带标签的评分数据DataFrame，在原评分数据基础上添加'tag'列
                包含每个用户-电影对的聚合标签文本
        tag_df: TF-IDF特征DataFrame，包含tag_0到tag_{n-1}的TF-IDF特征向量
               与rat_tag具有相同的索引，可直接合并
    
    Note:
        - 特征维度由config.tfidf_dim控制
        - 没有标签的用户-电影对将填充空字符串
        - TF-IDF特征能够捕捉标签的语义信息和重要性
        - 返回的两个DataFrame索引对齐，便于后续特征合并
    """
    logger.info(f"开始创建TF-IDF标签特征，目标维度: {config.tfidf_dim}")
    
    try:
        # 1. 按用户-电影对聚合标签
        logger.info("聚合用户-电影标签文档...")
        # 将同一用户对同一电影的所有标签合并为一个文档
        tags_grp = tags.groupby(['userId', 'movieId'])['tag'].apply(
            lambda seq: " ".join(seq.astype(str))
        ).reset_index()
        
        logger.info(f"聚合后标签记录数: {len(tags_grp)}，原始标签数: {len(tags)}")
        
        # 2. 与评分数据合并
        logger.info("合并标签与评分数据...")
        # 左连接确保保留所有评分记录，缺失标签填充为空字符串
        rat_tag = ratings.merge(tags_grp, on=['userId', 'movieId'], how='left')
        rat_tag = rat_tag.fillna({'tag': ''})
        
        # 统计标签覆盖情况
        has_tags = (rat_tag['tag'] != '').sum()
        total_ratings = len(rat_tag)
        coverage = has_tags / total_ratings * 100
        logger.info(f"标签覆盖率: {coverage:.1f}% ({has_tags}/{total_ratings})")
        
        # 3. 创建TF-IDF向量化器
        logger.info("初始化TF-IDF向量化器...")
        tfidf = TfidfVectorizer(
            max_features=config.tfidf_dim,  # 限制特征维度
            lowercase=True,  # 转换为小写
            stop_words='english',  # 过滤英文停用词
            ngram_range=(1, 2),  # 使用1-gram和2-gram
            min_df=2,  # 忽略出现次数少于2的词汇
            max_df=0.95  # 忽略出现频率超过95%的词汇
        )
        
        # 4. 拟合并转换标签文档
        logger.info("生成TF-IDF特征矩阵...")
        tfidf_mat = tfidf.fit_transform(rat_tag['tag']).toarray()
        
        logger.info(f"TF-IDF矩阵形状: {tfidf_mat.shape}")
        logger.info(f"词汇表大小: {len(tfidf.vocabulary_) if hasattr(tfidf, 'vocabulary_') else 0}")
        
        # 5. 转换为DataFrame格式
        logger.info("转换为DataFrame格式...")
        tag_df = pd.DataFrame(
            tfidf_mat, 
            columns=[f"tag_{i}" for i in range(tfidf_mat.shape[1])], 
            index=rat_tag.index
        )
        
        logger.info(f"TF-IDF标签特征创建完成:")
        logger.info(f"  评分+标签数据: {rat_tag.shape}")
        logger.info(f"  TF-IDF特征: {tag_df.shape}")
        logger.info(f"  平均非零特征数: {(tfidf_mat != 0).sum(axis=1).mean():.2f}")
        
        return rat_tag, tag_df
        
    except Exception as e:
        logger.error(f"TF-IDF标签特征创建失败: {e}")
        raise


def create_user_profile_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    创建用户画像特征
    
    该函数通过分析用户的历史评分行为，构建用户的个性化画像特征，
    包括评分统计特征和类型偏好特征，帮助模型理解用户的偏好模式。
    
    特征类型：
    1. 用户统计特征：评分均值、标准差等反映用户评分习惯的统计量
    2. 用户类型偏好：用户对不同电影类型的平均评分，反映类型偏好
    
    Args:
        ratings: 评分数据DataFrame，必须包含['userId', 'movieId', 'rating']列
        movies: 电影数据DataFrame，必须包含['movieId', 'genres']列
               genres应为已分割的列表格式
    
    Returns:
        user_stats: 用户统计特征DataFrame，包含：
                   - userId: 用户ID
                   - user_mean_rating: 用户平均评分
                   - user_std_rating: 用户评分标准差
        user_genre_pref: 用户类型偏好特征DataFrame，包含：
                        - userId: 用户ID
                        - user_genre_pref_{genre}: 用户对各类型的平均评分
    
    Note:
        - 评分标准差为0的用户（只有一个评分）会被填充为0
        - 用户未评分的类型会被填充为0
        - 类型偏好特征能够捕捉用户的类型倾向性
    """
    logger.info("开始创建用户画像特征")
    
    try:
        # 1. 创建用户统计特征
        logger.info("计算用户评分统计特征...")
        user_stats = ratings.groupby("userId").agg(
            user_mean_rating=('rating', 'mean'),  # 用户平均评分
            user_std_rating=('rating', 'std'),    # 用户评分标准差
            user_rating_count=('rating', 'count') # 用户评分数量
        ).reset_index()
        
        # 填充缺失的标准差（只有一个评分的用户）
        user_stats = user_stats.fillna({'user_std_rating': 0})
        
        logger.info(f"用户统计特征: {user_stats.shape}")
        logger.info(f"平均评分范围: {user_stats['user_mean_rating'].min():.2f} - {user_stats['user_mean_rating'].max():.2f}")
        logger.info(f"平均评分数量: {user_stats['user_rating_count'].mean():.1f}")
        
        # 2. 创建用户类型偏好特征
        logger.info("计算用户类型偏好特征...")
        # 合并评分和电影数据
        user_genre_data = ratings.merge(movies[['movieId', 'genres']], on='movieId')
        
        # 展开电影类型（每个类型一行）
        user_genre_data = user_genre_data.explode('genres')
        
        # 过滤掉空类型
        user_genre_data = user_genre_data[user_genre_data['genres'].notna()]
        user_genre_data = user_genre_data[user_genre_data['genres'] != '']
        
        # 计算用户对每个类型的平均评分
        user_genre_pref = user_genre_data.groupby(['userId', 'genres'])['rating'].mean().unstack()
        
        # 填充缺失值（用户未评分的类型）
        user_genre_pref = user_genre_pref.fillna(0)
        
        # 重命名列名
        user_genre_pref.columns = [f"user_genre_pref_{g}" for g in user_genre_pref.columns]
        user_genre_pref = user_genre_pref.reset_index()
        
        logger.info(f"用户类型偏好特征: {user_genre_pref.shape}")
        logger.info(f"识别类型数量: {len([col for col in user_genre_pref.columns if col.startswith('user_genre_pref_')])}")
        
        # 从用户统计特征中移除评分数量（通常不直接用作特征）
        user_stats = user_stats.drop('user_rating_count', axis=1)
        
        logger.info(f"用户画像特征创建完成:")
        logger.info(f"  统计特征: {user_stats.shape}")
        logger.info(f"  类型偏好特征: {user_genre_pref.shape}")
        
        return user_stats, user_genre_pref
        
    except Exception as e:
        logger.error(f"用户画像特征创建失败: {e}")
        raise


def create_movie_profile_features(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    创建电影画像特征
    
    该函数通过分析电影的历史评分数据，构建电影的统计画像特征，
    包括平均评分、评分数量等反映电影质量和受欢迎程度的指标。
    
    特征说明：
    1. 平均评分：反映电影的整体质量水平
    2. 评分数量：反映电影的受关注程度和流行度
    3. 评分标准差：反映用户对电影评价的一致性
    
    Args:
        ratings: 评分数据DataFrame，必须包含['movieId', 'rating']列
    
    Returns:
        movie_stats: 电影统计特征DataFrame，包含：
                    - movieId: 电影ID
                    - movie_avg_rating: 电影平均评分
                    - movie_num_ratings: 电影评分数量
                    - movie_rating_std: 电影评分标准差
    
    Note:
        - 只有一个评分的电影，标准差会被填充为0
        - 这些特征有助于模型理解电影的受欢迎程度和质量
        - 评分数量可以作为置信度的指标
    """
    logger.info("开始创建电影画像特征")
    
    try:
        # 计算电影统计特征
        logger.info("计算电影评分统计特征...")
        movie_stats = ratings.groupby("movieId").agg(
            movie_avg_rating=('rating', 'mean'),   # 电影平均评分
            movie_num_ratings=('rating', 'count'), # 电影评分数量
            movie_rating_std=('rating', 'std')     # 电影评分标准差
        ).reset_index()
        
        # 填充缺失的标准差（只有一个评分的电影）
        movie_stats = movie_stats.fillna({'movie_rating_std': 0})
        
        # 记录统计信息
        logger.info(f"电影画像特征创建完成: {movie_stats.shape}")
        logger.info(f"电影平均评分范围: {movie_stats['movie_avg_rating'].min():.2f} - {movie_stats['movie_avg_rating'].max():.2f}")
        logger.info(f"电影评分数量范围: {movie_stats['movie_num_ratings'].min()} - {movie_stats['movie_num_ratings'].max()}")
        logger.info(f"平均每部电影评分数: {movie_stats['movie_num_ratings'].mean():.1f}")
        
        return movie_stats
        
    except Exception as e:
        logger.error(f"电影画像特征创建失败: {e}")
        raise


def merge_features(ratings: pd.DataFrame, movies_feats: pd.DataFrame, user_f: pd.DataFrame, 
                  item_f: pd.DataFrame, user_bias: pd.DataFrame, item_bias: pd.DataFrame,
                  user_stats: pd.DataFrame, user_genre_pref: pd.DataFrame, 
                  movie_stats: pd.DataFrame, tag_df: pd.DataFrame, 
                  rat_tag: pd.DataFrame) -> pd.DataFrame:
    """
    合并所有特征并生成交叉特征
    
    该函数将各个模块生成的不同类型特征进行合并，形成完整的特征矩阵，
    同时生成用户和物品隐因子的交叉特征，为机器学习模型提供丰富的输入。
    
    合并顺序和策略：
    1. 以带标签的评分数据为基础
    2. 依次左连接各类特征（保留所有评分记录）
    3. 生成用户-物品隐因子交叉特征
    4. 处理缺失值
    
    Args:
        ratings: 原始评分数据DataFrame
        movies_feats: 电影内容特征（年份、类型等）
        user_f: 用户隐因子特征
        item_f: 电影隐因子特征
        user_bias: 用户偏置特征
        item_bias: 电影偏置特征
        user_stats: 用户统计特征
        user_genre_pref: 用户类型偏好特征
        movie_stats: 电影统计特征
        tag_df: TF-IDF标签特征
        rat_tag: 带标签的评分数据
    
    Returns:
        df: 合并后的完整特征DataFrame，包含：
           - 原始评分信息
           - 所有类型的特征
           - 用户-物品交叉特征
    
    Note:
        - 使用左连接确保保留所有评分记录
        - 交叉特征通过元素级乘法生成
        - 缺失特征会被适当填充
    """
    logger.info("开始合并所有特征")
    
    try:
        # 1. 以带标签的评分数据为基础
        logger.info("以评分+标签数据为基础...")
        df = rat_tag.copy()
        initial_shape = df.shape
        logger.info(f"基础数据形状: {initial_shape}")
        
        # 2. 依次合并各类特征
        logger.info("合并电影内容特征...")
        df = df.merge(movies_feats, on='movieId', how='left')
        logger.info(f"合并后形状: {df.shape}")
        
        logger.info("合并用户隐因子特征...")
        df = df.merge(user_f, on='userId', how='left')
        logger.info(f"合并后形状: {df.shape}")
        
        logger.info("合并电影隐因子特征...")
        df = df.merge(item_f, on='movieId', how='left')
        logger.info(f"合并后形状: {df.shape}")
        
        logger.info("合并用户偏置特征...")
        df = df.merge(user_bias, on='userId', how='left')
        logger.info(f"合并后形状: {df.shape}")
        
        logger.info("合并电影偏置特征...")
        df = df.merge(item_bias, on='movieId', how='left')
        logger.info(f"合并后形状: {df.shape}")
        
        logger.info("合并用户统计特征...")
        df = df.merge(user_stats, on='userId', how='left')
        logger.info(f"合并后形状: {df.shape}")
        
        logger.info("合并用户类型偏好特征...")
        df = df.merge(user_genre_pref, on='userId', how='left')
        logger.info(f"合并后形状: {df.shape}")
        
        logger.info("合并电影统计特征...")
        df = df.merge(movie_stats, on='movieId', how='left')
        logger.info(f"合并后形状: {df.shape}")
        
        logger.info("合并TF-IDF标签特征...")
        # 使用concat而不是merge，因为tag_df与df具有相同的索引
        df = pd.concat([df, tag_df], axis=1)
        logger.info(f"合并后形状: {df.shape}")
        
        # 3. 生成交叉特征
        logger.info(f"生成{config.latent_dim}个用户-物品交叉特征...")
        cross_features = []
        for i in range(config.latent_dim):
            # 用户隐因子与物品隐因子的元素级乘积
            cross_feature = df[f"user_f{i}"] * df[f"item_f{i}"]
            df[f"cross_f{i}"] = cross_feature
            cross_features.append(f"cross_f{i}")
        
        logger.info(f"交叉特征生成完成: {cross_features}")
        
        # 4. 检查缺失值情况
        missing_counts = df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        if len(missing_features) > 0:
            logger.warning(f"发现缺失值的特征: {len(missing_features)}个")
            for feature, count in missing_features.head(10).items():
                logger.warning(f"  {feature}: {count}个缺失值")
        
        # 5. 记录最终统计信息
        final_shape = df.shape
        feature_count = final_shape[1] - initial_shape[1]
        
        logger.info(f"特征合并完成:")
        logger.info(f"  最终数据形状: {final_shape}")
        logger.info(f"  新增特征数: {feature_count}")
        logger.info(f"  总特征数: {final_shape[1]}")
        logger.info(f"  数据完整性: {(1 - df.isnull().sum().sum() / df.size) * 100:.1f}%")
        
        return df
        
    except Exception as e:
        logger.error(f"特征合并失败: {e}")
        raise