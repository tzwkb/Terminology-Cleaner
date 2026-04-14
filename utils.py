"""
工具函数模块
包含文本处理、语言检测、问题分类、数据验证等功能
"""

import pandas as pd
import re
import string
from typing import Tuple, Dict
from logger import get_logger

logger = get_logger()


def is_acronym_or_code(text):
    """
    判断文本是否是缩写、型号或代号
    
    特征：
    - 全大写字母（如 NATO, SWIFT, IRIS）
    - 包含数字（如 K-15, F-117, X-43）
    - 包含连字符、斜杠等（如 Pantsir-S1E, DIME/WEX）
    - 长度较短（≤15字符）
    - 混合大小写但首字母大写且较短（如 Puma, Mica）
    
    参数:
        text: 待判断的文本
    返回:
        True表示可能是缩写/型号，False表示普通词汇
    """
    if pd.isna(text):
        return False
    
    text = str(text).strip()
    
    # 空文本
    if not text:
        return False
    
    # 长度超过15个字符，不太可能是缩写
    if len(text) > 15:
        return False
    
    # 包含数字（如 K-15, F-117, X-43）
    if any(c.isdigit() for c in text):
        return True
    
    # 包含连字符、斜杠等特殊符号（如 Pantsir-S1E, DIME/WEX）
    if any(c in text for c in ['-', '/', '_', '+']):
        return True
    
    # 全大写字母（如 NATO, SWIFT, IRIS）
    if text.isupper() and text.isalpha():
        return True
    
    # 短词且首字母大写（如 Puma, Mica）- 避免误判单字母
    if len(text) > 1 and len(text) <= 6 and text[0].isupper():
        return True
    
    return False


def normalize_text(text, config):
    """
    标准化文本用于比较
    
    参数:
        text: 原始文本
        config: 清洗配置
    返回:
        标准化后的文本
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # 去除首尾标点符号
    if config.remove_punctuation:
        # 定义中英文标点符号
        punctuation = string.punctuation + '。，、；：？！…—·《》〈〉「」『』【】〔〕""''（）'
        text = text.strip(punctuation)
    
    # 大小写标准化
    if config.ignore_case:
        text = text.lower()
    
    return text.strip()


def detect_language_detailed(text) -> Dict[str, any]:
    """
    详细检测文本的语言组成
    
    参数:
        text: 待检测的文本
    
    返回:
        字典，包含：
        - language: 'zh' (中文), 'en' (英文), 'mixed' (混合), 'other' (其他)
        - chinese_chars: 中文字符数
        - english_chars: 英文字符数
        - japanese_chars: 日语字符数
        - korean_chars: 韩语字符数
        - digit_chars: 数字字符数
        - total_chars: 总有效字符数
        - is_pure: 是否是纯单一语言
    """
    if pd.isna(text) or not text:
        return {
            'language': 'other',
            'chinese_chars': 0,
            'english_chars': 0,
            'japanese_chars': 0,
            'korean_chars': 0,
            'digit_chars': 0,
            'total_chars': 0,
            'is_pure': False
        }
    
    text = str(text).strip()
    if not text:
        return {
            'language': 'other',
            'chinese_chars': 0,
            'english_chars': 0,
            'japanese_chars': 0,
            'korean_chars': 0,
            'digit_chars': 0,
            'total_chars': 0,
            'is_pure': False
        }
    
    # 统计各类字符
    chinese_chars = 0  # 中文汉字
    english_chars = 0  # 英文字母
    japanese_chars = 0  # 日语假名
    korean_chars = 0   # 韩语字符
    digit_chars = 0    # 数字
    
    for char in text:
        # 中文汉字 (CJK统一汉字)
        if '\u4e00' <= char <= '\u9fff':
            chinese_chars += 1
        # 英文字母 (ASCII)
        elif char.isalpha() and ord(char) < 128:
            english_chars += 1
        # 日语平假名
        elif '\u3040' <= char <= '\u309f':
            japanese_chars += 1
        # 日语片假名
        elif '\u30a0' <= char <= '\u30ff':
            japanese_chars += 1
        # 韩语
        elif '\uac00' <= char <= '\ud7af' or '\u1100' <= char <= '\u11ff':
            korean_chars += 1
        # 数字
        elif char.isdigit():
            digit_chars += 1
    
    # 总有效字符数（语言字符 + 数字，不包括标点和空格）
    total_chars = chinese_chars + english_chars + japanese_chars + korean_chars + digit_chars
    
    if total_chars == 0:
        return {
            'language': 'other',
            'chinese_chars': 0,
            'english_chars': 0,
            'japanese_chars': 0,
            'korean_chars': 0,
            'digit_chars': 0,
            'total_chars': 0,
            'is_pure': False
        }
    
    # 计算比例
    chinese_ratio = chinese_chars / total_chars
    english_ratio = english_chars / total_chars
    japanese_ratio = japanese_chars / total_chars
    korean_ratio = korean_chars / total_chars
    
    # 判断主要语言
    language = 'other'
    is_pure = False
    
    # 纯中文（允许少量数字）
    if chinese_ratio > 0.8 and english_chars == 0:
        language = 'zh'
        is_pure = True
    # 纯英文（允许少量数字）
    elif english_ratio > 0.8 and chinese_chars == 0:
        language = 'en'
        is_pure = True
    # 中英混合
    elif chinese_chars > 0 and english_chars > 0:
        language = 'mixed'
        is_pure = False
    # 中文为主
    elif chinese_ratio > 0.3:
        language = 'zh'
        is_pure = chinese_ratio > 0.8
    # 英文为主
    elif english_ratio > 0.5:
        language = 'en'
        is_pure = english_ratio > 0.8
    # 日语
    elif japanese_ratio > 0.3:
        language = 'ja'
        is_pure = japanese_ratio > 0.8
    # 韩语
    elif korean_ratio > 0.3:
        language = 'ko'
        is_pure = korean_ratio > 0.8
    else:
        language = 'other'
        is_pure = False
    
    return {
        'language': language,
        'chinese_chars': chinese_chars,
        'english_chars': english_chars,
        'japanese_chars': japanese_chars,
        'korean_chars': korean_chars,
        'digit_chars': digit_chars,
        'total_chars': total_chars,
        'is_pure': is_pure
    }


def detect_language(text):
    """
    检测文本的主要语言（兼容旧接口）
    
    返回: 'zh' (中文), 'en' (英文), 'mixed' (混合), 'ja' (日语), 'ko' (韩语), 'other' (其他)
    """
    result = detect_language_detailed(text)
    return result['language']


def detect_column_language(column_name):
    """
    根据列名检测该列应该是什么语言
    
    返回: 'zh', 'en', 'unknown'
    """
    column_lower = str(column_name).lower()
    
    # 中文相关
    if any(keyword in column_lower for keyword in ['cn', 'zh', 'chinese', '中文', '中', '汉语']):
        return 'zh'
    # 英文相关
    elif any(keyword in column_lower for keyword in ['en', 'english', '英文', '英']):
        return 'en'
    else:
        return 'unknown'


def contains_chinese(text):
    """
    检查文本是否包含中文字符
    
    参数:
        text: 待检查的文本
    返回:
        True表示包含中文字符，False表示不包含
    """
    if pd.isna(text):
        return False
    
    text = str(text).strip()
    # 检查是否包含中文字符（包括中文标点）
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def categorize_issue(issue_description):
    """
    将问题描述归类到问题大类和具体问题
    
    返回: (问题大类, 具体问题)
    """
    # 定义问题分类映射（按优先级从高到低排序）
    # 先检查更具体的问题类型，再检查宽泛的关键词
    if '相似' in issue_description and ('原文相似' in issue_description or '译文相似' in issue_description or '相似术语' in issue_description):
        return '相似术语', issue_description
    elif '冲突' in issue_description:
        return '翻译冲突', issue_description
    elif '重复的术语对' in issue_description or '互为翻译' in issue_description or '反向重复' in issue_description:
        return '重复性问题', issue_description
    elif '原文与译文' in issue_description and '一致' in issue_description:
        return '一致性问题', issue_description
    elif '语言顺序' in issue_description or '语言错误' in issue_description or '包含中文字符' in issue_description:
        return '语言顺序问题', issue_description
    elif '数字不一致' in issue_description:
        return '数据一致性问题', issue_description
    elif '长度比' in issue_description or '过长' in issue_description:
        return '长度异常', issue_description
    elif '过短' in issue_description or '纯数字' in issue_description or '特殊字符' in issue_description:
        return '格式问题', issue_description
    elif '空值' in issue_description or re.search(r'\bnan\b', issue_description, re.IGNORECASE):
        # 使用正则表达式匹配独立的"nan"单词，避免误判"Maintenance"等包含"nan"的正常词汇
        return '数据完整性', issue_description
    else:
        return '其他问题', issue_description


def get_severity_level(category, issue_description):
    """
    根据问题大类和具体描述确定严重程度
    
    返回: "严重" | "警告" | "提示"
    """
    # 严重问题：必须处理
    severe_categories = {
        '数据完整性',      # 空值、nan
        '重复性问题',      # 重复条目
        '语言顺序问题',    # 语言错误
        '数据一致性问题',  # 数字不一致
    }
    
    # 警告问题：建议检查
    warning_categories = {
        '翻译冲突',        # 一对多、多对一
        '长度异常',        # 长度比例异常
        '一致性问题',      # 原文译文一致（可能是缩写）
        '相似术语',        # 高度相似的术语（可能是拼写错误或同义词）
    }
    
    # 提示问题：可选检查
    # info_categories = {
    #     '格式问题',
    #     '其他问题',
    # }
    
    # 复合问题：根据具体问题描述判断
    if category == '复合问题':
        # 检查是否包含严重问题的关键词
        severe_keywords = ['空值', 'nan', '重复', '语言错误', '中文字符', '数字不一致', '纯英文']
        if any(keyword in issue_description for keyword in severe_keywords):
            return '严重'
        # 检查是否包含警告问题的关键词
        warning_keywords = ['冲突', '长度比', '过长', '一致']
        if any(keyword in issue_description for keyword in warning_keywords):
            return '警告'
        return '提示'
    
    if category in severe_categories:
        return '严重'
    elif category in warning_categories:
        return '警告'
    else:
        return '提示'


# ============================================================================
# 数据验证函数
# ============================================================================

def extract_numbers(text):
    """
    从文本中提取所有数字（包括整数、小数、负数）
    
    参数:
        text: 待提取的文本
    返回:
        数字集合
    """
    if pd.isna(text):
        return set()
    
    text = str(text)
    # 匹配整数、小数、负数
    numbers = re.findall(r'-?\d+\.?\d*', text)
    # 转换为浮点数并去重
    return set(float(n) for n in numbers if n and n != '-')


def check_number_consistency(source_text, target_text):
    """
    检查原文和译文中的数字是否一致
    
    参数:
        source_text: 原文
        target_text: 译文
    返回:
        (是否一致, 差异描述)
    """
    source_numbers = extract_numbers(source_text)
    target_numbers = extract_numbers(target_text)
    
    # 如果两边都没有数字，认为一致
    if not source_numbers and not target_numbers:
        return True, None
    
    # 如果只有一边有数字，可能是正常的（如单位转换、补充说明等）
    # 所以只检查两边都有数字的情况
    if not source_numbers or not target_numbers:
        return True, None
    
    # 检查数字是否完全一致
    if source_numbers == target_numbers:
        return True, None
    
    # 数字不一致
    only_in_source = source_numbers - target_numbers
    only_in_target = target_numbers - source_numbers
    
    desc_parts = []
    if only_in_source:
        source_str = ', '.join(str(int(n) if n.is_integer() else n) for n in sorted(only_in_source))
        desc_parts.append(f"原文有: {source_str}")
    if only_in_target:
        target_str = ', '.join(str(int(n) if n.is_integer() else n) for n in sorted(only_in_target))
        desc_parts.append(f"译文有: {target_str}")
    
    return False, '; '.join(desc_parts)


def check_length_ratio(source_text, target_text):
    """
    检查原文和译文的长度比例是否异常
    
    参数:
        source_text: 原文
        target_text: 译文
    返回:
        (是否正常, 异常描述)
    """
    if pd.isna(source_text) or pd.isna(target_text):
        return True, None
    
    source_len = len(str(source_text).strip())
    target_len = len(str(target_text).strip())
    
    # 至少一个长度为0
    if source_len == 0 or target_len == 0:
        return True, None
    
    # 计算长度比例（较长/较短）
    ratio = max(source_len, target_len) / min(source_len, target_len)
    
    # 比例阈值：
    # - 如果比例 > 10，说明一方过长或过短，可能有问题
    # - 但对于很短的术语（<5字符），放宽到20倍
    min_len = min(source_len, target_len)
    threshold = 20 if min_len < 5 else 10
    
    if ratio > threshold:
        if source_len > target_len:
            return False, f"原文过长（长度比 {ratio:.1f}:1，原文{source_len}字符 vs 译文{target_len}字符）"
        else:
            return False, f"译文过长（长度比 {ratio:.1f}:1，译文{target_len}字符 vs 原文{source_len}字符）"
    
    return True, None

