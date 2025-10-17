import pandas as pd
import os
import re
import string
from datetime import datetime

# 相似度检测相关导入（可选）
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class CleanConfig:
    """清洗配置类"""
    def __init__(self):
        self.ignore_case = True  # 是否忽略大小写
        self.remove_punctuation = True  # 是否去除首尾标点
        self.quality_check = True  # 是否进行质量检查
        self.interactive_mode = False  # 是否交互式确认互为翻译
        self.similarity_check = False  # 是否进行相似术语检测
        self.similarity_threshold = 0.85  # 相似度阈值（0-1）
        self.char_sim_weight = 0.4  # 字符相似度权重
        self.semantic_sim_weight = 0.6  # 语义相似度权重
        
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


def detect_language(text):
    """
    检测文本的主要语言
    
    返回: 'zh' (中文), 'en' (英文), 'mixed' (混合), 'other' (其他)
    """
    if pd.isna(text) or not text:
        return 'other'
    
    text = str(text).strip()
    if not text:
        return 'other'
    
    # 统计中文字符
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    # 统计英文字母
    english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    # 总字符数（排除空格和标点）
    total_chars = sum(1 for c in text if c.isalnum())
    
    if total_chars == 0:
        return 'other'
    
    chinese_ratio = chinese_chars / total_chars
    english_ratio = english_chars / total_chars
    
    # 判断语言
    if chinese_ratio > 0.3:
        return 'zh'
    elif english_ratio > 0.5:
        return 'en'
    elif chinese_ratio > 0.1 and english_ratio > 0.1:
        return 'mixed'
    else:
        return 'other'


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


def categorize_issue(issue_description):
    """
    将问题描述归类到问题大类和具体问题
    
    返回: (问题大类, 具体问题)
    """
    # 定义问题分类映射（按优先级从高到低排序）
    # 先检查更具体的问题类型，再检查宽泛的关键词
    if '冲突' in issue_description:
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


class SimilarityDetector:
    """
    相似术语检测器
    使用混合方法：字符串相似度 + 语义相似度
    """
    def __init__(self, config=None):
        self.config = config or CleanConfig()
        self.semantic_model = None
        
        # 检查依赖
        if not RAPIDFUZZ_AVAILABLE:
            print("警告：未安装 rapidfuzz，字符相似度将使用较慢的备用方法")
            print("建议安装：pip install rapidfuzz")
        
        if self.config.similarity_check and not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("警告：未安装 sentence-transformers，无法使用语义相似度")
            print("安装方法：pip install sentence-transformers")
            self.config.similarity_check = False
    
    def load_semantic_model(self):
        """延迟加载语义模型（第一次使用时才加载）"""
        if self.semantic_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            print("正在加载语义相似度模型...")
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("模型加载完成")
    
    def char_similarity(self, term1, term2):
        """
        计算字符串相似度（0-1）
        """
        if pd.isna(term1) or pd.isna(term2):
            return 0.0
        
        term1 = str(term1).strip().lower()
        term2 = str(term2).strip().lower()
        
        if not term1 or not term2:
            return 0.0
        
        if RAPIDFUZZ_AVAILABLE:
            # 使用 rapidfuzz（快速）
            return fuzz.ratio(term1, term2) / 100.0
        else:
            # 备用方法：使用 difflib
            from difflib import SequenceMatcher
            return SequenceMatcher(None, term1, term2).ratio()
    
    def semantic_similarity(self, term1, term2):
        """
        计算语义相似度（0-1）
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.semantic_model is None:
            return 0.0
        
        if pd.isna(term1) or pd.isna(term2):
            return 0.0
        
        term1 = str(term1).strip()
        term2 = str(term2).strip()
        
        if not term1 or not term2:
            return 0.0
        
        # 编码并计算余弦相似度
        emb1 = self.semantic_model.encode(term1, convert_to_tensor=True)
        emb2 = self.semantic_model.encode(term2, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2)[0][0])
    
    def hybrid_similarity(self, term1, term2):
        """
        混合相似度计算（自适应权重）
        
        返回: {
            'final': 最终相似度,
            'char': 字符相似度,
            'semantic': 语义相似度,
            'method': 使用的方法
        }
        """
        char_sim = self.char_similarity(term1, term2)
        
        # 如果只有字符相似度可用
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.semantic_model is None:
            return {
                'final': char_sim,
                'char': char_sim,
                'semantic': None,
                'method': '仅字符相似度'
            }
        
        semantic_sim = self.semantic_similarity(term1, term2)
        
        # 自适应权重策略
        if char_sim > 0.90:
            # 字符高度相似（>90%），优先字符相似度（可能是拼写问题）
            final_sim = 0.8 * char_sim + 0.2 * semantic_sim
            method = '高字符相似'
        elif char_sim < 0.30:
            # 字符差异大（<30%），优先语义相似度（可能是同义词）
            final_sim = 0.2 * char_sim + 0.8 * semantic_sim
            method = '低字符相似'
        else:
            # 中等情况，使用配置的权重
            final_sim = (self.config.char_sim_weight * char_sim + 
                        self.config.semantic_sim_weight * semantic_sim)
            method = '平衡权重'
        
        return {
            'final': final_sim,
            'char': char_sim,
            'semantic': semantic_sim,
            'method': method
        }
    
    def detect_similar_terms(self, terms, column_name='术语'):
        """
        两阶段检测相似术语
        
        参数:
            terms: 术语列表
            column_name: 列名（用于报告）
        
        返回:
            相似术语对列表
        """
        if not terms or len(terms) < 2:
            return []
        
        # 去重和清洗
        unique_terms = list(set([str(t).strip() for t in terms if pd.notna(t) and str(t).strip()]))
        
        if len(unique_terms) < 2:
            return []
        
        print(f"\n正在检测 {column_name} 列的相似术语...")
        print(f"  术语总数: {len(unique_terms)}")
        
        similar_pairs = []
        
        # 第一阶段：字符相似度快速筛选
        print("  阶段1: 字符相似度筛选...")
        candidates = []
        
        for i, term1 in enumerate(unique_terms):
            for term2 in unique_terms[i+1:]:
                char_sim = self.char_similarity(term1, term2)
                
                # 字符相似度 > 50% 的进入第二阶段
                if char_sim > 0.50:
                    candidates.append((term1, term2, char_sim))
        
        print(f"  找到 {len(candidates)} 对候选相似术语")
        
        if not candidates:
            return []
        
        # 第二阶段：语义相似度精确计算
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("  阶段2: 语义相似度计算...")
            self.load_semantic_model()
            
            for term1, term2, char_sim in candidates:
                result = self.hybrid_similarity(term1, term2)
                
                # 最终相似度超过阈值
                if result['final'] >= self.config.similarity_threshold:
                    similar_pairs.append({
                        '列名': column_name,
                        '术语1': term1,
                        '术语2': term2,
                        '相似度': f"{result['final']*100:.1f}%",
                        '字符相似度': f"{result['char']*100:.1f}%",
                        '语义相似度': f"{result['semantic']*100:.1f}%" if result['semantic'] is not None else 'N/A',
                        '检测方法': result['method']
                    })
        else:
            # 只使用字符相似度
            for term1, term2, char_sim in candidates:
                if char_sim >= self.config.similarity_threshold:
                    similar_pairs.append({
                        '列名': column_name,
                        '术语1': term1,
                        '术语2': term2,
                        '相似度': f"{char_sim*100:.1f}%",
                        '字符相似度': f"{char_sim*100:.1f}%",
                        '语义相似度': 'N/A',
                        '检测方法': '仅字符相似度'
                    })
        
        print(f"  发现 {len(similar_pairs)} 对高度相似术语")
        
        return similar_pairs


def clean_terminology_table(input_file, output_clean=None, output_removed=None, output_uncertain=None, config=None):
    """
    清洗双列双语术语表
    
    参数:
        input_file: 输入的Excel或CSV文件路径
        output_clean: 清洗后的术语表输出路径（默认：原文件名_cleaned.xlsx/.csv）
        output_removed: 被删除条目的输出路径（默认：原文件名_removed.xlsx/.csv）
        output_uncertain: 不确定条目的输出路径（默认：原文件名_uncertain.xlsx/.csv）
        config: 清洗配置对象（CleanConfig实例）
    """
    
    # 使用默认配置
    if config is None:
        config = CleanConfig()
    
    # 读取文件（支持CSV和Excel）
    print(f"正在读取文件: {input_file}")
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(input_file)
    elif file_ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}。请使用CSV或Excel文件。")
    
    # 检查列数
    if len(df.columns) < 2:
        print("错误：文件至少需要两列（原文和译文）")
        return
    
    # 使用前两列作为原文和译文
    source_col = df.columns[0]
    target_col = df.columns[1]
    
    print(f"原文列: {source_col}")
    print(f"译文列: {target_col}")
    print(f"总条目数: {len(df)}")
    print(f"\n清洗配置:")
    print(f"  - 忽略大小写: {'是' if config.ignore_case else '否'}")
    print(f"  - 去除标点: {'是' if config.remove_punctuation else '否'}")
    print(f"  - 质量检查: {'是' if config.quality_check else '否'}")
    
    # 创建删除条目列表和不确定条目列表
    removed_data = []
    uncertain_data = []
    
    # 记录原始索引
    df_with_index = df.copy()
    df_with_index['原始行号'] = df.index + 2  # Excel行号从2开始（1是标题）
    
    # 1. 标记包含"nan"的行（检查字符串形式的"nan"和实际的NaN值）
    mask_nan = pd.Series(False, index=df_with_index.index)
    
    for idx, row in df_with_index.iterrows():
        source_val = row[source_col]
        target_val = row[target_col]
        
        # 检查是否为NaN
        is_source_nan = pd.isna(source_val)
        is_target_nan = pd.isna(target_val)
        
        # 检查字符串是否为空值（完全匹配，不会误判包含"nan"的正常单词如"nanotechnology"）
        contains_nan_str = False
        source_str = str(source_val).strip() if not is_source_nan else ''
        target_str = str(target_val).strip() if not is_target_nan else ''
        
        # 定义空值列表：只有完全匹配这些值才认为是空值（大小写不敏感）
        # 注意：不包含 'na' 以避免误判 'nanotechnology' 等正常词汇
        empty_values = ['nan', 'NaN', 'NAN', '', 'null', 'NULL', 'Null', 'none', 'None', 'NONE', 
                        'n/a', 'N/A', '#n/a', '#N/A', '#na', '#NA']
        if not is_source_nan and source_str in empty_values:
            contains_nan_str = True
        if not is_target_nan and target_str in empty_values:
            contains_nan_str = True
            
        if is_source_nan or is_target_nan or contains_nan_str:
            mask_nan.loc[idx] = True
            issue_desc = '包含空值或nan'
            category, detail = categorize_issue(issue_desc)
            removed_data.append({
                '原始行号': row['原始行号'],
                source_col: source_val,
                target_col: target_val,
                '问题大类': category,
                '具体问题': detail
            })
    
    # 先移除含nan的行
    df_no_nan = df_with_index[~mask_nan].copy()
    
    # 2. 标记原文和译文完全一致的行（智能判断缩写/型号）
    print("\n正在检查原文与译文一致的条目...")
    mask_identical = pd.Series(False, index=df_no_nan.index)
    
    for idx, row in df_no_nan.iterrows():
        source_normalized = normalize_text(row[source_col], config)
        target_normalized = normalize_text(row[target_col], config)
        source_original = str(row[source_col]).strip()
        target_original = str(row[target_col]).strip()
        
        if source_normalized == target_normalized and source_normalized != '':
            # 判断是否是缩写/型号/代号
            if is_acronym_or_code(source_original):
                # 是缩写/型号：标记为不确定条目，但保留在数据中（不删除）
                issue_desc = '原文与译文一致（可能是缩写/型号）'
                category, detail = categorize_issue(issue_desc)
                uncertain_data.append({
                    '原始行号': row['原始行号'],
                    source_col: source_original,
                    target_col: target_original,
                    '问题大类': category,
                    '具体问题': detail,
                    '冲突组ID': ''  # 保持与其他uncertain_data格式一致
                })
            else:
                # 普通词汇：删除
                mask_identical.loc[idx] = True
                issue_desc = '原文与译文完全一致（非缩写）'
                category, detail = categorize_issue(issue_desc)
                removed_data.append({
                    '原始行号': row['原始行号'],
                    source_col: source_original,
                    target_col: target_original,
                    '问题大类': category,
                    '具体问题': detail
                })
    
    # 移除原文和译文一致的普通词汇（缩写/型号已保留）
    df_no_identical = df_no_nan[~mask_identical].copy()
    
    # 3. 去除重复的术语对（保留第一次出现）
    print("\n正在检查重复的术语对...")
    # 使用标准化文本进行比较
    df_no_identical['source_cleaned'] = df_no_identical[source_col].apply(lambda x: normalize_text(x, config))
    df_no_identical['target_cleaned'] = df_no_identical[target_col].apply(lambda x: normalize_text(x, config))
    
    # 找出重复项
    duplicates = df_no_identical.duplicated(subset=['source_cleaned', 'target_cleaned'], keep='first')
    
    for idx, row in df_no_identical[duplicates].iterrows():
        issue_desc = '重复的术语对'
        category, detail = categorize_issue(issue_desc)
        removed_data.append({
            '原始行号': row['原始行号'],
            source_col: row[source_col],
            target_col: row[target_col],
            '问题大类': category,
            '具体问题': detail
        })
    
    # 移除重复项
    df_no_duplicates = df_no_identical[~duplicates].copy()
    
    # 4. 识别互为翻译的术语对（A→B 和 B→A）
    print("\n正在检查互为翻译的术语对...")
    mask_reverse = pd.Series(False, index=df_no_duplicates.index)
    
    # 检测列的期望语言
    source_expected_lang = detect_column_language(source_col)
    target_expected_lang = detect_column_language(target_col)
    
    print(f"  - {source_col} 列期望语言: {source_expected_lang}")
    print(f"  - {target_col} 列期望语言: {target_expected_lang}")
    
    # 创建一个字典来快速查找已出现的术语对及其索引
    seen_pairs = {}  # (source, target) -> idx
    reverse_pairs_to_confirm = []  # 需要确认的互为翻译对
    
    for idx, row in df_no_duplicates.iterrows():
        source_val = row['source_cleaned']
        target_val = row['target_cleaned']
        
        # 检查反向术语对是否已经存在
        if (target_val, source_val) in seen_pairs:
            # 找到互为翻译的术语对
            first_idx = seen_pairs[(target_val, source_val)]
            first_row = df_no_duplicates.loc[first_idx]
            
            if config.interactive_mode:
                # 交互式模式：让用户选择保留哪一条
                reverse_pairs_to_confirm.append({
                    'first_idx': first_idx,
                    'first_row': first_row,
                    'second_idx': idx,
                    'second_row': row
                })
            else:
                # 自动模式：智能选择保留语言顺序正确的那一条
                # 检测第一条的语言匹配情况
                first_source_lang = detect_language(first_row[source_col])
                first_target_lang = detect_language(first_row[target_col])
                first_match_score = 0
                if source_expected_lang != 'unknown' and first_source_lang == source_expected_lang:
                    first_match_score += 1
                if target_expected_lang != 'unknown' and first_target_lang == target_expected_lang:
                    first_match_score += 1
                
                # 检测第二条的语言匹配情况
                second_source_lang = detect_language(row[source_col])
                second_target_lang = detect_language(row[target_col])
                second_match_score = 0
                if source_expected_lang != 'unknown' and second_source_lang == source_expected_lang:
                    second_match_score += 1
                if target_expected_lang != 'unknown' and second_target_lang == target_expected_lang:
                    second_match_score += 1
                
                # 根据匹配分数决定删除哪一条
                if second_match_score > first_match_score:
                    # 第二条语言顺序更正确，删除第一条
                    mask_reverse.loc[first_idx] = True
                    issue_desc = '互为翻译的术语对（语言顺序错误）'
                    category, detail = categorize_issue(issue_desc)
                    removed_data.append({
                        '原始行号': first_row['原始行号'],
                        source_col: first_row[source_col],
                        target_col: first_row[target_col],
                        '问题大类': category,
                        '具体问题': detail
                    })
                else:
                    # 第一条语言顺序更正确或相同，删除第二条
                    mask_reverse.loc[idx] = True
                    issue_desc = '互为翻译的术语对（语言顺序错误）' if first_match_score > second_match_score else '互为翻译的术语对（反向重复）'
                    category, detail = categorize_issue(issue_desc)
                    removed_data.append({
                        '原始行号': row['原始行号'],
                        source_col: row[source_col],
                        target_col: row[target_col],
                        '问题大类': category,
                        '具体问题': detail
                    })
        else:
            # 将当前术语对添加到已见字典
            seen_pairs[(source_val, target_val)] = idx
    
    # 处理交互式确认的互为翻译对
    if config.interactive_mode and reverse_pairs_to_confirm:
        print(f"\n发现 {len(reverse_pairs_to_confirm)} 对互为翻译的术语，请选择保留哪一条：")
        for i, pair in enumerate(reverse_pairs_to_confirm, 1):
            first_row = pair['first_row']
            second_row = pair['second_row']
            print(f"\n第 {i} 对:")
            print(f"  A) 行 {first_row['原始行号']}: {first_row[source_col]} → {first_row[target_col]}")
            print(f"  B) 行 {second_row['原始行号']}: {second_row[source_col]} → {second_row[target_col]}")
            
            while True:
                choice = input("  保留哪一条？[A/B/都保留(K)] > ").strip().upper()
                if choice in ['A', 'B', 'K']:
                    break
                print("  无效输入，请输入 A、B 或 K")
            
            if choice == 'B':
                # 删除第一条，保留第二条
                mask_reverse.loc[pair['first_idx']] = True
                issue_desc = '互为翻译的术语对（用户选择删除）'
                category, detail = categorize_issue(issue_desc)
                removed_data.append({
                    '原始行号': first_row['原始行号'],
                    source_col: first_row[source_col],
                    target_col: first_row[target_col],
                    '问题大类': category,
                    '具体问题': detail
                })
            elif choice == 'A':
                # 删除第二条，保留第一条
                mask_reverse.loc[pair['second_idx']] = True
                issue_desc = '互为翻译的术语对（用户选择删除）'
                category, detail = categorize_issue(issue_desc)
                removed_data.append({
                    '原始行号': second_row['原始行号'],
                    source_col: second_row[source_col],
                    target_col: second_row[target_col],
                    '问题大类': category,
                    '具体问题': detail
                })
            # choice == 'K' 时都保留，不做任何操作
    
    # 移除互为翻译的项
    df_cleaned = df_no_duplicates[~mask_reverse].copy()
    
    # 5. 筛选翻译冲突：相同原文对应不同译文，或不同原文对应相同译文
    print("\n正在检查翻译冲突...")
    
    # 使用字典来跟踪：原文 -> [译文列表]，译文 -> [原文列表]
    source_to_targets = {}  # {source: [(target, row_num, source_orig, target_orig)]}
    target_to_sources = {}  # {target: [(source, row_num, source_orig, target_orig)]}
    
    for idx, row in df_cleaned.iterrows():
        source_cleaned = row['source_cleaned']
        target_cleaned = row['target_cleaned']
        source_original = row[source_col]
        target_original = row[target_col]
        row_num = row['原始行号']
        
        # 记录原文到译文的映射
        if source_cleaned not in source_to_targets:
            source_to_targets[source_cleaned] = []
        source_to_targets[source_cleaned].append((target_cleaned, row_num, source_original, target_original))
        
        # 记录译文到原文的映射
        if target_cleaned not in target_to_sources:
            target_to_sources[target_cleaned] = []
        target_to_sources[target_cleaned].append((source_cleaned, row_num, source_original, target_original))
    
    # 找出相同原文对应多个不同译文的情况
    source_conflicts = {}
    for source, targets_list in source_to_targets.items():
        unique_targets = set([t[0] for t in targets_list])
        if len(unique_targets) > 1:
            source_conflicts[source] = targets_list
    
    # 找出相同译文对应多个不同原文的情况
    target_conflicts = {}
    for target, sources_list in target_to_sources.items():
        unique_sources = set([s[0] for s in sources_list])
        if len(unique_sources) > 1:
            target_conflicts[target] = sources_list
    
    # 生成冲突报告
    if source_conflicts or target_conflicts:
        print(f"  发现 {len(source_conflicts)} 个原文冲突（一对多）")
        print(f"  发现 {len(target_conflicts)} 个译文冲突（多对一）")
        
        # 处理原文冲突（一个原文对应多个不同译文）
        conflict_group_id = 1
        for source, targets_list in source_conflicts.items():
            unique_targets = set([t[0] for t in targets_list])
            issue_desc = f"原文冲突（一对多）：原文 '{targets_list[0][2]}' 有 {len(unique_targets)} 个不同译文"
            category, _ = categorize_issue(issue_desc)
            
            for target, row_num, source_orig, target_orig in targets_list:
                uncertain_data.append({
                    '原始行号': row_num,
                    source_col: source_orig,
                    target_col: target_orig,
                    '问题大类': category,
                    '具体问题': issue_desc,
                    '冲突组ID': f"G{conflict_group_id}"
                })
            conflict_group_id += 1
        
        # 处理译文冲突（多个不同原文对应一个译文）
        for target, sources_list in target_conflicts.items():
            unique_sources = set([s[0] for s in sources_list])
            issue_desc = f"译文冲突（多对一）：译文 '{sources_list[0][3]}' 有 {len(unique_sources)} 个不同原文"
            category, _ = categorize_issue(issue_desc)
            
            for source, row_num, source_orig, target_orig in sources_list:
                uncertain_data.append({
                    '原始行号': row_num,
                    source_col: source_orig,
                    target_col: target_orig,
                    '问题大类': category,
                    '具体问题': issue_desc,
                    '冲突组ID': f"G{conflict_group_id}"
                })
            conflict_group_id += 1
    else:
        print("  未发现翻译冲突")
    
    # 6. 质量检查：标记可疑条目（不删除，单独输出）
    if config.quality_check:
        print("\n正在进行质量检查...")
        for idx, row in df_cleaned.iterrows():
            source_val = str(row[source_col]).strip()
            target_val = str(row[target_col]).strip()
            reasons = []
            
            # 检查语言是否符合规则：
            # - 英文术语中不能出现中文字符（严格）
            # - 中文术语中可以出现英文字符（允许混合）
            
            # 如果原文列期望是英文，检查是否包含中文字符
            if source_expected_lang == 'en' and contains_chinese(source_val):
                reasons.append("原文语言错误（英文术语中包含中文字符）")
            
            # 如果译文列期望是英文，检查是否包含中文字符
            if target_expected_lang == 'en' and contains_chinese(target_val):
                reasons.append("译文语言错误（英文术语中包含中文字符）")
            
            # 如果原文列期望是中文，检查是否完全是英文（不包含任何中文）
            if source_expected_lang == 'zh':
                source_actual_lang = detect_language(source_val)
                if source_actual_lang == 'en':
                    reasons.append("原文语言错误（期望中文，实际纯英文）")
            
            # 如果译文列期望是中文，检查是否完全是英文（不包含任何中文）
            if target_expected_lang == 'zh':
                target_actual_lang = detect_language(target_val)
                if target_actual_lang == 'en':
                    reasons.append("译文语言错误（期望中文，实际纯英文）")
            
            # 检查数字一致性
            is_number_consistent, number_diff = check_number_consistency(source_val, target_val)
            if not is_number_consistent:
                reasons.append(f"数字不一致（{number_diff}）")
            
            # 检查长度比例
            is_length_normal, length_issue = check_length_ratio(source_val, target_val)
            if not is_length_normal:
                reasons.append(length_issue)
            
            # 检查过短
            if len(source_val) <= 1:
                reasons.append("原文过短(≤1字符)")
            if len(target_val) <= 1:
                reasons.append("译文过短(≤1字符)")
            
            # 检查过长
            if len(source_val) > 100:
                reasons.append(f"原文过长({len(source_val)}字符)")
            if len(target_val) > 100:
                reasons.append(f"译文过长({len(target_val)}字符)")
            
            # 检查纯数字
            if source_val.isdigit():
                reasons.append("原文为纯数字")
            if target_val.isdigit():
                reasons.append("译文为纯数字")
            
            # 检查特殊字符比例（>30%被认为过多）
            source_special_ratio = sum(1 for c in source_val if not c.isalnum() and not c.isspace()) / len(source_val) if source_val else 0
            target_special_ratio = sum(1 for c in target_val if not c.isalnum() and not c.isspace()) / len(target_val) if target_val else 0
            
            if source_special_ratio > 0.3:
                reasons.append(f"原文特殊字符过多({source_special_ratio*100:.0f}%)")
            if target_special_ratio > 0.3:
                reasons.append(f"译文特殊字符过多({target_special_ratio*100:.0f}%)")
            
            if reasons:
                issue_desc = '; '.join(reasons)
                category, detail = categorize_issue(issue_desc)
                uncertain_data.append({
                    '原始行号': row['原始行号'],
                    source_col: source_val,
                    target_col: target_val,
                    '问题大类': category,
                    '具体问题': detail,
                    '冲突组ID': ''  # 质量问题没有冲突组
                })
    
    # 从 df_cleaned 中移除所有有问题的条目（包括 uncertain_data）
    if uncertain_data:
        uncertain_row_numbers = set(item['原始行号'] for item in uncertain_data)
        df_cleaned = df_cleaned[~df_cleaned['原始行号'].isin(uncertain_row_numbers)].copy()
        print(f"\n  从清洗后文件中移除 {len(uncertain_data)} 个不确定条目")
    
    # 删除临时列
    df_cleaned = df_cleaned.drop(columns=['原始行号', 'source_cleaned', 'target_cleaned'])
    
    # 生成输出文件名（统一处理）
    base_name = os.path.splitext(input_file)[0]
    if output_clean is None:
        output_clean = f"{base_name}_cleaned{file_ext}"
    if output_removed is None:
        output_removed = f"{base_name}_removed{file_ext}"
    if output_uncertain is None:
        output_uncertain = f"{base_name}_uncertain{file_ext}"  # 保留参数但不使用
    
    # 保存清洗后的数据（根据文件格式选择保存方式）
    if file_ext == '.csv':
        df_cleaned.to_csv(output_clean, index=False, encoding='utf-8-sig')
    else:
        df_cleaned.to_excel(output_clean, index=False)
    print(f"\n✓ 清洗后的术语表已保存: {output_clean}")
    print(f"  保留条目数: {len(df_cleaned)} ({len(df_cleaned)/len(df)*100:.2f}%)")
    print(f"  注：此文件只包含完全没有问题的条目")
    
    # 计算详细统计信息
    stats = {
        '原文字符数': df_cleaned[source_col].astype(str).str.len(),
        '译文字符数': df_cleaned[target_col].astype(str).str.len()
    }
    
    # 合并 removed_data 和 uncertain_data，并添加严重程度
    # 策略：
    # 1. 同一术语的多个非冲突问题合并成一条（严重程度取最高）
    # 2. 翻译冲突保持独立显示（需要显示冲突组信息）
    
    all_removed_data = []
    
    # 处理明确删除的条目
    for item in removed_data:
        item['冲突组ID'] = ''  # 明确删除的没有冲突组
        # 添加严重程度
        item['严重程度'] = get_severity_level(item['问题大类'], item['具体问题'])
        all_removed_data.append(item)
    
    # 处理不确定条目（需要审核的）- 按原始行号分组
    uncertain_by_row = {}
    for item in uncertain_data:
        if '冲突组ID' not in item:
            item['冲突组ID'] = ''
        # 添加严重程度
        item['严重程度'] = get_severity_level(item['问题大类'], item['具体问题'])
        
        row_num = item['原始行号']
        if row_num not in uncertain_by_row:
            uncertain_by_row[row_num] = []
        uncertain_by_row[row_num].append(item)
    
    # 合并同一行的非冲突问题
    for row_num, items in uncertain_by_row.items():
        # 分离冲突和非冲突问题
        conflict_items = [item for item in items if item['冲突组ID']]
        non_conflict_items = [item for item in items if not item['冲突组ID']]
        
        # 翻译冲突问题保持独立（需要显示冲突组）
        all_removed_data.extend(conflict_items)
        
        # 非冲突问题合并成一条
        if non_conflict_items:
            if len(non_conflict_items) == 1:
                # 只有一个质量问题，直接添加
                all_removed_data.append(non_conflict_items[0])
            else:
                # 多个质量问题，合并（理论上不会发生，因为质量检查已经合并了）
                merged_item = non_conflict_items[0].copy()
                
                # 合并所有问题大类（去重）
                all_categories = list(set(item['问题大类'] for item in non_conflict_items))
                
                # 合并所有具体问题
                all_issues = [item['具体问题'] for item in non_conflict_items]
                merged_item['具体问题'] = '; '.join(all_issues)
                
                # 如果有多个问题大类，使用"复合问题"
                if len(all_categories) > 1:
                    merged_item['问题大类'] = '复合问题'
                else:
                    merged_item['问题大类'] = all_categories[0]
                
                # 严重程度取最高的（数值最小的）
                severity_order = {'严重': 0, '警告': 1, '提示': 2}
                severities = [item['严重程度'] for item in non_conflict_items]
                merged_item['严重程度'] = min(severities, key=lambda x: severity_order[x])
                
                all_removed_data.append(merged_item)
    
    # 保存所有问题条目到 removed 文件
    if all_removed_data:
        df_removed = pd.DataFrame(all_removed_data)
        
        # 重新排列列顺序：严重程度、原始行号、问题大类放在前面
        cols = ['严重程度', '原始行号', '问题大类'] + [col for col in df_removed.columns if col not in ['严重程度', '原始行号', '问题大类']]
        df_removed = df_removed[cols]
        
        # 按严重程度、问题大类、冲突组ID、原始行号排序
        # 严重程度排序：严重 > 警告 > 提示
        severity_order = {'严重': 0, '警告': 1, '提示': 2}
        df_removed['_severity_order'] = df_removed['严重程度'].map(severity_order)
        df_removed = df_removed.sort_values(['_severity_order', '问题大类', '冲突组ID', '原始行号'])
        df_removed = df_removed.drop(columns=['_severity_order'])
        
        if file_ext == '.csv':
            df_removed.to_csv(output_removed, index=False, encoding='utf-8-sig')
        else:
            df_removed.to_excel(output_removed, index=False)
        
        print(f"\n✓ 问题条目已保存: {output_removed}")
        print(f"  问题条目总数: {len(df_removed)} ({len(df_removed)/len(df)*100:.2f}%)")
        
        # 按严重程度统计
        print("\n按严重程度统计:")
        severity_counts = df_removed['严重程度'].value_counts()
        for severity in ['严重', '警告', '提示']:  # 按固定顺序输出
            if severity in severity_counts:
                count = severity_counts[severity]
                print(f"  - {severity}: {count}条 ({count/len(df)*100:.2f}%)")
        
        # 按问题大类统计
        print("\n按问题大类统计:")
        category_counts = df_removed.groupby(['问题大类', '严重程度']).size()
        for (category, severity), count in category_counts.items():
            print(f"  - {category} [{severity}]: {count}条 ({count/len(df)*100:.2f}%)")
    else:
        print("\n✓ 没有发现问题条目")
    
    # 7. 相似术语检测（可选）
    similar_terms_data = []
    if config.similarity_check:
        print("\n" + "="*60)
        print("相似术语检测")
        print("="*60)
        
        detector = SimilarityDetector(config)
        
        # 分别检测原文和译文列
        source_similar = detector.detect_similar_terms(
            df_cleaned[source_col].tolist(), 
            column_name=source_col
        )
        target_similar = detector.detect_similar_terms(
            df_cleaned[target_col].tolist(), 
            column_name=target_col
        )
        
        similar_terms_data = source_similar + target_similar
        
        # 保存相似术语报告
        if similar_terms_data:
            output_similar = f"{base_name}_similar{file_ext}"
            df_similar = pd.DataFrame(similar_terms_data)
            
            if file_ext == '.csv':
                df_similar.to_csv(output_similar, index=False, encoding='utf-8-sig')
            else:
                df_similar.to_excel(output_similar, index=False)
            
            print(f"\n✓ 相似术语报告已保存: {output_similar}")
            print(f"  发现 {len(similar_terms_data)} 对相似术语")
            print(f"  建议人工检查是否为重复、拼写错误或同义词")
        else:
            print("\n✓ 未发现高度相似的术语")
    
    # 生成清洗报告
    total_removed = len(all_removed_data) if all_removed_data else 0
    
    report = f"""
==================== 术语表清洗报告 ====================
清洗时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
输入文件: {input_file}

清洗配置:
  - 忽略大小写: {'是' if config.ignore_case else '否'}
  - 去除标点: {'是' if config.remove_punctuation else '否'}
  - 质量检查: {'是' if config.quality_check else '否'}
  - 相似术语检测: {'是' if config.similarity_check else '否'}

基本统计:
  - 原始条目数: {len(df)}
  - 清洗后条目数: {len(df_cleaned)} ({len(df_cleaned)/len(df)*100:.2f}%)
    注：此文件只包含完全没有问题的条目
  - 问题条目数: {total_removed} ({total_removed/len(df)*100:.2f}%)
    注：所有问题条目已移至 removed 文件，供人工审核

术语长度统计（清洗后）:
  - 原文平均长度: {stats['原文字符数'].mean():.1f} 字符
  - 原文最短: {stats['原文字符数'].min()} 字符
  - 原文最长: {stats['原文字符数'].max()} 字符
  - 译文平均长度: {stats['译文字符数'].mean():.1f} 字符
  - 译文最短: {stats['译文字符数'].min()} 字符
  - 译文最长: {stats['译文字符数'].max()} 字符

问题条目按严重程度统计:
"""
    if all_removed_data:
        df_removed_report = pd.DataFrame(all_removed_data)
        severity_counts = df_removed_report['严重程度'].value_counts()
        for severity in ['严重', '警告', '提示']:
            if severity in severity_counts:
                count = severity_counts[severity]
                report += f"  - {severity}: {count}条 ({count/len(df)*100:.2f}%)\n"
        
        report += "\n问题条目按问题大类统计:\n"
        category_counts = df_removed_report.groupby(['问题大类', '严重程度']).size()
        for (category, severity), count in category_counts.items():
            report += f"  - {category} [{severity}]: {count}条 ({count/len(df)*100:.2f}%)\n"
    else:
        report += "  无问题条目\n"
    
    if similar_terms_data:
        report += f"\n相似术语检测:\n"
        report += f"  - 发现相似术语对: {len(similar_terms_data)} 对\n"
        report += f"  - 阈值设置: {config.similarity_threshold*100:.0f}%\n"
        report += f"  - 建议人工检查是否为重复、拼写错误或同义词\n"
    
    report += f"""
输出文件:
  - 清洗后术语表: {output_clean}
  - 问题条目表: {output_removed}"""
    
    if similar_terms_data:
        report += f"\n  - 相似术语表: {base_name}_similar{file_ext}"
    
    report += """

说明:
  - cleaned 文件：完全干净的术语表，可直接使用
  - removed 文件：所有问题条目，按严重程度排序
    * 严重：必须处理（空值、重复、语言错误、数字不一致）
    * 警告：建议检查（翻译冲突、长度异常）
    * 提示：可选检查（格式问题等）"""
    
    if similar_terms_data:
        report += """
  - similar 文件：高度相似的术语对
    * 可能是拼写错误、同义词或重复术语
    * 建议人工审核并统一"""
    
    report += """
=======================================================
"""
    
    print(report)
    
    # 保存报告
    report_file = f"{os.path.splitext(input_file)[0]}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ 清洗报告已保存: {report_file}")
    
    return {
        'cleaned_count': len(df_cleaned),
        'removed_count': total_removed,
        'success': True
    }


def batch_process_directory(directory, config=None):
    """
    批量处理目录下的所有术语表文件
    
    参数:
        directory: 目录路径
        config: 清洗配置对象
    """
    if config is None:
        config = CleanConfig()
    
    # 递归查找所有支持的文件（包括子文件夹）
    import glob
    patterns = ['**/*.csv', '**/*.xlsx', '**/*.xls']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    
    if not files:
        print(f"错误：目录 {directory} 中没有找到CSV或Excel文件")
        return
    
    print("=" * 60)
    print(f"批量处理模式（递归搜索）")
    print("=" * 60)
    print(f"在目录 '{directory}' 及其子目录中找到 {len(files)} 个文件:")
    for i, file in enumerate(files, 1):
        # 显示相对路径
        rel_path = os.path.relpath(file, directory)
        print(f"  {i}. {rel_path}")
    
    print(f"\n开始处理...\n")
    
    results = []
    for i, file in enumerate(files, 1):
        rel_path = os.path.relpath(file, directory)
        print(f"\n{'='*60}")
        print(f"处理文件 {i}/{len(files)}: {rel_path}")
        print(f"{'='*60}")
        
        try:
            result = clean_terminology_table(file, config=config)
            result['file'] = rel_path  # 使用相对路径
            results.append(result)
        except Exception as e:
            print(f"✗ 处理失败: {str(e)}")
            results.append({
                'file': rel_path,  # 使用相对路径
                'success': False,
                'error': str(e)
            })
    
    # 生成批量处理报告
    print("\n" + "=" * 60)
    print("批量处理总结")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.get('success', False))
    fail_count = len(results) - success_count
    
    print(f"\n总文件数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    
    if success_count > 0:
        total_cleaned = sum(r.get('cleaned_count', 0) for r in results if r.get('success', False))
        total_removed = sum(r.get('removed_count', 0) for r in results if r.get('success', False))
        
        print(f"\n总计:")
        print(f"  - 保留条目（cleaned）: {total_cleaned}")
        print(f"  - 问题条目（removed）: {total_removed}")
    
    if fail_count > 0:
        print(f"\n失败的文件:")
        for r in results:
            if not r.get('success', False):
                print(f"  - {r['file']}: {r.get('error', '未知错误')}")
    
    # 保存批量处理报告
    batch_report_file = os.path.join(directory, f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(batch_report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("批量处理报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"目录: {directory}\n\n")
        
        for r in results:
            f.write(f"\n文件: {r['file']}\n")
            if r.get('success', False):
                f.write(f"  状态: 成功\n")
                f.write(f"  保留条目（cleaned）: {r.get('cleaned_count', 0)}\n")
                f.write(f"  问题条目（removed）: {r.get('removed_count', 0)}\n")
            else:
                f.write(f"  状态: 失败\n")
                f.write(f"  错误: {r.get('error', '未知错误')}\n")
    
    print(f"\n✓ 批量处理报告已保存: {batch_report_file}")


def main():
    """
    主函数：处理命令行参数或交互式输入
    """
    import sys
    
    # 创建配置对象
    config = CleanConfig()
    
    if len(sys.argv) > 1:
        # 命令行模式
        input_path = sys.argv[1]
        
        # 检查是否为批量处理
        if os.path.isdir(input_path):
            batch_process_directory(input_path, config)
            return
        
        output_clean = sys.argv[2] if len(sys.argv) > 2 else None
        output_removed = sys.argv[3] if len(sys.argv) > 3 else None
        output_uncertain = sys.argv[4] if len(sys.argv) > 4 else None
    else:
        # 交互式模式
        print("=" * 60)
        print("双语术语表清洗工具 v2.0")
        print("=" * 60)
        
        # 选择处理模式
        print("\n处理模式:")
        print("  1. 单文件处理")
        print("  2. 批量处理（处理整个目录）")
        mode = input("\n请选择模式 [1/2，默认1]: ").strip() or "1"
        
        if mode == "2":
            # 批量处理模式
            print("\n请输入目录路径（留空使用当前目录）:")
            directory = input("> ").strip() or "."
            
            if not os.path.isdir(directory):
                print(f"错误：目录不存在: {directory}")
                return
            
            # 配置选项
            print("\n=== 清洗配置 ===")
            config.ignore_case = input("忽略大小写差异？[Y/n，默认Y]: ").strip().lower() != 'n'
            config.remove_punctuation = input("去除首尾标点？[Y/n，默认Y]: ").strip().lower() != 'n'
            config.quality_check = input("进行质量检查？[Y/n，默认Y]: ").strip().lower() != 'n'
            config.similarity_check = input("检测相似术语？[y/N，默认N]: ").strip().lower() == 'y'
            
            if config.similarity_check:
                threshold_input = input(f"相似度阈值（0-100，默认{config.similarity_threshold*100:.0f}）: ").strip()
                if threshold_input:
                    try:
                        threshold = float(threshold_input) / 100
                        if 0 <= threshold <= 1:
                            config.similarity_threshold = threshold
                    except ValueError:
                        print("  警告：阈值无效，使用默认值")
            
            batch_process_directory(directory, config)
            return
        
        # 单文件处理模式
        # 列出当前目录的Excel和CSV文件
        data_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls', '.csv'))]
        
        if data_files:
            print("\n当前目录下的Excel/CSV文件:")
            for i, file in enumerate(data_files, 1):
                print(f"  {i}. {file}")
            print("\n请输入文件路径或序号:")
            user_input = input("> ").strip()
            
            # 检查是否为序号
            if user_input.isdigit() and 1 <= int(user_input) <= len(data_files):
                input_path = data_files[int(user_input) - 1]
            else:
                input_path = user_input
        else:
            print("\n请输入Excel或CSV文件路径:")
            input_path = input("> ").strip()
        
        # 配置选项
        print("\n=== 清洗配置 ===")
        config.ignore_case = input("忽略大小写差异？[Y/n，默认Y]: ").strip().lower() != 'n'
        config.remove_punctuation = input("去除首尾标点？[Y/n，默认Y]: ").strip().lower() != 'n'
        config.quality_check = input("进行质量检查？[Y/n，默认Y]: ").strip().lower() != 'n'
        config.similarity_check = input("检测相似术语？[y/N，默认N]: ").strip().lower() == 'y'
        config.interactive_mode = input("互为翻译时交互式确认？[y/N，默认N]: ").strip().lower() == 'y'
        
        if config.similarity_check:
            threshold_input = input(f"相似度阈值（0-100，默认{config.similarity_threshold*100:.0f}）: ").strip()
            if threshold_input:
                try:
                    threshold = float(threshold_input) / 100
                    if 0 <= threshold <= 1:
                        config.similarity_threshold = threshold
                except ValueError:
                    print("  警告：阈值无效，使用默认值")
        
        output_clean = None
        output_removed = None
        output_uncertain = None
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：文件不存在: {input_path}")
        return
    
    # 执行清洗
    try:
        clean_terminology_table(input_path, output_clean, output_removed, output_uncertain, config)
        print("\n✓ 清洗完成！")
    except Exception as e:
        print(f"\n✗ 清洗过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

