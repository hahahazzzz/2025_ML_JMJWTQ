#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字体配置模块

专门用于解决matplotlib中文字体显示问题的配置模块。

核心功能：
1. 跨平台中文字体自动检测和配置
2. 系统字体优先级管理
3. 项目自定义字体支持
4. 字体显示效果测试
5. 字体加载异常处理

支持平台：
- macOS: PingFang HK, Hiragino Sans GB, STHeiti, Arial Unicode MS
- Windows: Microsoft YaHei, SimHei, KaiTi, FangSong
- Linux: Noto Sans CJK SC, WenQuanYi Micro Hei, Source Han Sans SC

字体优先级：
1. 项目自定义字体（fonts/NotoSansSC-Regular.otf）
2. 系统优质中文字体
3. matplotlib默认字体

技术特点：
- 自动检测系统可用字体
- 动态加载项目字体文件
- 完整的异常处理和降级机制
- 字体显示效果验证
- 跨平台兼容性保证

使用方式：
- 在可视化模块中导入并调用setup_chinese_fonts()
- 支持手动获取字体属性对象
- 提供字体显示测试功能
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties, fontManager
import platform
import os
from pathlib import Path
try:
    from utils.logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def setup_chinese_fonts():
    """
    设置matplotlib的中文字体显示
    
    自动检测当前系统平台，选择最适合的中文字体进行配置。
    优先使用项目自定义字体，然后按优先级尝试系统字体。
    
    配置内容：
    - 设置font.family为sans-serif
    - 配置font.sans-serif字体列表
    - 禁用unicode负号显示问题
    - 验证字体可用性
    - 加载项目自定义字体（如存在）
    
    异常处理：
    - 字体检测失败时使用默认配置
    - 项目字体加载失败时记录警告
    - 确保基本的中文显示功能
    
    Note:
        该函数应在创建任何matplotlib图表之前调用
    """
    try:
        # 根据系统选择合适的中文字体
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            # 使用系统检测到的可用字体
            fonts = ['PingFang HK', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
        elif system == 'Windows':
            fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
        else:  # Linux
            fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Source Han Sans SC']
        
        # 设置字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = fonts + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 验证字体设置
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        found_font = None
        for font in fonts:
            if font in available_fonts:
                found_font = font
                break
        
        if found_font:
            logger.info(f"成功设置中文字体: {found_font}")
        else:
            logger.warning("未找到合适的中文字体，可能出现乱码")
            
        # 尝试加载项目字体作为备用
        current_dir = Path(__file__).parent.parent
        font_file = current_dir / 'fonts' / 'NotoSansSC-Regular.otf'
        
        if font_file.exists():
            try:
                fm.fontManager.addfont(str(font_file))
                font_prop = FontProperties(fname=str(font_file))
                font_name = font_prop.get_name()
                
                # 将项目字体添加到字体列表前面
                current_fonts = plt.rcParams['font.sans-serif']
                plt.rcParams['font.sans-serif'] = [font_name] + [f for f in current_fonts if f != font_name]
                
                logger.info(f"已加载项目字体: {font_name}")
            except Exception as e:
                logger.warning(f"项目字体加载失败: {e}")
            
    except Exception as e:
        logger.error(f"字体设置失败: {e}")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False


def get_chinese_font():
    """
    获取中文字体属性对象
    
    按优先级尝试获取可用的中文字体，返回FontProperties对象。
    可用于需要显式指定字体的matplotlib组件。
    
    字体查找顺序：
    1. 项目自定义字体文件
    2. 系统平台优质中文字体
    3. matplotlib默认字体
    
    Returns:
        FontProperties: 字体属性对象，可直接用于matplotlib组件
        
    Example:
        >>> font_prop = get_chinese_font()
        >>> plt.text(0.5, 0.5, '中文文本', fontproperties=font_prop)
        
    Note:
        即使所有字体加载失败，也会返回默认FontProperties对象
    """
    try:
        # 首先尝试项目字体
        current_dir = Path(__file__).parent.parent
        font_file = current_dir / 'fonts' / 'NotoSansSC-Regular.otf'
        
        if font_file.exists():
            return FontProperties(fname=str(font_file))
        
        # 尝试系统字体
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/Hiragino Sans GB.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc'
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return FontProperties(fname=font_path)
        
        elif system == 'Windows':
            font_paths = [
                'C:/Windows/Fonts/msyh.ttc',  # Microsoft YaHei
                'C:/Windows/Fonts/simhei.ttf',  # SimHei
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return FontProperties(fname=font_path)
        
        # 返回默认字体
        return FontProperties()
        
    except Exception as e:
        logger.error(f"获取字体失败: {e}")
        return FontProperties()


def test_font_display():
    """
    测试中文字体显示效果
    
    创建一个测试图表来验证中文字体是否正确显示。
    包含常见的中文文本内容，用于检查字体配置是否成功。
    
    测试内容：
    - 图表标题中文显示
    - 坐标轴标签中文显示
    - 文本内容中文显示
    - 不同字体大小和样式
    
    输出：
    - 在output目录生成font_test.png测试图片
    - 控制台输出测试结果信息
    - 日志记录详细的测试过程
    
    用途：
    - 验证字体配置是否成功
    - 检查中文显示是否正常
    - 调试字体相关问题
    
    Note:
        测试图片保存在项目的output目录下
    """
    try:
        import matplotlib.pyplot as plt
        
        # 设置字体
        setup_chinese_fonts()
        
        # 创建测试图表
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 测试中文显示
        test_texts = [
            '电影推荐系统可视化测试',
            '预测评分分布图',
            '特征重要性排名',
            '误差分析结果'
        ]
        
        for i, text in enumerate(test_texts):
            ax.text(0.1, 0.8 - i * 0.15, text, fontsize=14, 
                   transform=ax.transAxes)
        
        ax.set_title('中文字体显示测试', fontsize=16, fontweight='bold')
        ax.set_xlabel('横轴标签测试', fontsize=12)
        ax.set_ylabel('纵轴标签测试', fontsize=12)
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 保存测试图片
        current_dir = Path(__file__).parent.parent
        test_path = current_dir / 'output' / 'font_test.png'
        test_path.parent.mkdir(exist_ok=True)
        
        plt.savefig(test_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"字体测试图片已保存到: {test_path}")
        print(f"字体测试完成，请查看图片: {test_path}")
        
    except Exception as e:
        logger.error(f"字体测试失败: {e}")
        print(f"字体测试失败: {e}")


if __name__ == '__main__':
    # 设置基本日志
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 运行字体测试
    test_font_display()