"""
核心清洗模块
包含术语表清洗的主要逻辑
"""

import pandas as pd
import os
import glob
from datetime import datetime

from config import CleanConfig
from logger import get_logger
from utils import (
    normalize_text, detect_language, detect_column_language,
    categorize_issue, get_severity_level, contains_chinese, is_acronym_or_code,
    check_number_consistency, check_length_ratio
)
from similarity_detector import SimilarityDetector

logger = get_logger()


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
    logger.info(f"正在读取文件: {input_file}")
    file_ext = os.path.splitext(input_file)[1].lower()
    
    try:
        if file_ext == '.csv':
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            for encoding in encodings:
                try:
                    df = pd.read_csv(input_file, encoding=encoding)
                    logger.debug(f"成功使用编码 {encoding} 读取CSV文件")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"无法识别CSV文件编码: {input_file}")
        elif file_ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
            df = pd.read_excel(input_file, engine='openpyxl' if file_ext == '.xlsx' else None)
        else:
            error_msg = f"不支持的文件格式: {file_ext}。请使用CSV或Excel文件。"
            logger.error(error_msg)
            raise ValueError(error_msg)
    except Exception as e:
        logger.error(f"文件读取失败: {str(e)}", exc_info=True)
        raise
    
    # 检查列数
    if len(df.columns) < 2:
        error_msg = "文件至少需要两列（原文和译文）"
        logger.error(error_msg)
        print(f"错误：{error_msg}")
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
            
            # 如果原文列期望是中文，检查是否完全不含中文（纯英文或其他）
            if source_expected_lang == 'zh':
                # 只有完全不包含中文字符才算纯英文
                if not contains_chinese(source_val):
                    reasons.append("原文语言错误（期望中文，实际无中文字符）")
            
            # 如果译文列期望是中文，检查是否完全不含中文（纯英文或其他）
            if target_expected_lang == 'zh':
                # 只有完全不包含中文字符才算纯英文
                if not contains_chinese(target_val):
                    reasons.append("译文语言错误（期望中文，实际无中文字符）")
            
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
    
    # 7. 相似术语检测（可选）- 辅助发现潜在问题
    if config.similarity_check:
        print("\n正在检测相似术语（可能的重复/拼写错误）...")
        
        detector = SimilarityDetector(config)
        
        # 检测原文列的相似术语
        source_similar = detector.detect_similar_terms(
            df_cleaned[source_col].tolist(), 
            column_name=source_col
        )
        
        # 检测译文列的相似术语
        target_similar = detector.detect_similar_terms(
            df_cleaned[target_col].tolist(), 
            column_name=target_col
        )
        
        # 将相似术语作为质量问题添加到不确定条目
        all_similar = source_similar + target_similar
        
        if all_similar:
            print(f"  发现 {len(all_similar)} 对相似术语")
            
            # 优化：预先创建清理后的列（避免重复计算）
            source_cleaned = df_cleaned[source_col].astype(str).str.strip()
            target_cleaned = df_cleaned[target_col].astype(str).str.strip()
            col_cleaned_cache = {
                source_col: source_cleaned,
                target_col: target_cleaned
            }
            
            # 为每对相似术语创建问题条目（使用聚类分组）
            # 第一步：使用并查集将相似术语聚类
            from collections import defaultdict
            
            # 并查集：找到根节点
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                if parent[x] != x:
                    parent[x] = find(parent[x])  # 路径压缩
                return parent[x]
            
            def union(x, y):
                root_x = find(x)
                root_y = find(y)
                if root_x != root_y:
                    parent[root_x] = root_y
            
            # 合并所有相似的术语
            for similar_pair in all_similar:
                term1 = similar_pair['术语1']
                term2 = similar_pair['术语2']
                union(term1, term2)
            
            # 按根节点分组
            clusters = defaultdict(list)
            for similar_pair in all_similar:
                term1 = similar_pair['术语1']
                root = find(term1)
                clusters[root].append(similar_pair)
            
            # 为每个聚类分配组ID
            similarity_group_id = 1
            cluster_to_group_id = {}
            for root in clusters:
                cluster_to_group_id[root] = f"S{similarity_group_id}"
                similarity_group_id += 1
            
            # 统计聚类信息
            total_terms_in_clusters = len(set([similar_pair['术语1'] for similar_pair in all_similar] + 
                                              [similar_pair['术语2'] for similar_pair in all_similar]))
            print(f"  聚类结果：{total_terms_in_clusters}个相似术语归入{len(clusters)}个相似组")
            
            # 现在处理每对相似术语，使用聚类的组ID
            for similar_pair in all_similar:
                col_name = similar_pair['列名']
                term1 = similar_pair['术语1']
                term2 = similar_pair['术语2']
                similarity = similar_pair['相似度']
                
                # 判断是原文列还是译文列
                col_type = '原文' if col_name == source_col else '译文'
                
                # 获取这对术语所属的聚类组ID
                root = find(term1)
                similar_group_id = cluster_to_group_id[root]
                
                # 使用预先计算的清理列
                col_cleaned = col_cleaned_cache[col_name]
                
                # 查找term1的所有匹配行
                mask1 = col_cleaned == term1
                term1_df = df_cleaned[mask1]
                term1_pairs = [{
                    'row_num': row['原始行号'],
                    'source': row[source_col],
                    'target': row[target_col]
                } for _, row in term1_df.iterrows()] if len(term1_df) > 0 else []
                
                # 查找term2的所有匹配行
                mask2 = col_cleaned == term2
                term2_df = df_cleaned[mask2]
                term2_pairs = [{
                    'row_num': row['原始行号'],
                    'source': row[source_col],
                    'target': row[target_col]
                } for _, row in term2_df.iterrows()] if len(term2_df) > 0 else []
                
                # 为term1的所有匹配行添加问题
                for term1_info in term1_pairs:
                    # 取term2的第一个作为代表（如果有多个）
                    if term2_pairs:
                        term2_info = term2_pairs[0]
                        # 简化版：更清晰的格式
                        issue_desc = f"{col_type}相似 {similarity}：'{term1}' ↔ '{term2}' [对比术语对: {term2_info['source']} → {term2_info['target']}]"
                    else:
                        issue_desc = f"{col_type}相似 {similarity}：'{term1}' ↔ '{term2}'"
                    
                    category, detail = categorize_issue(issue_desc)
                    
                    # 检查是否已经存在该行的相似问题
                    already_exists = False
                    for item in uncertain_data:
                        if item['原始行号'] == term1_info['row_num'] and '相似' in item['具体问题']:
                            already_exists = True
                            break
                    
                    if not already_exists:
                        uncertain_data.append({
                            '原始行号': term1_info['row_num'],
                            source_col: term1_info['source'],
                            target_col: term1_info['target'],
                            '问题大类': category,
                            '具体问题': detail,
                            '冲突组ID': similar_group_id  # 使用相似组ID
                        })
                
                # 为term2的所有匹配行添加问题
                for term2_info in term2_pairs:
                    if term1_pairs:
                        term1_info = term1_pairs[0]
                        # 简化版：更清晰的格式
                        issue_desc = f"{col_type}相似 {similarity}：'{term2}' ↔ '{term1}' [对比术语对: {term1_info['source']} → {term1_info['target']}]"
                    else:
                        issue_desc = f"{col_type}相似 {similarity}：'{term2}' ↔ '{term1}'"
                    
                    category, detail = categorize_issue(issue_desc)
                    
                    # 检查是否已经存在该行的相似问题
                    already_exists = False
                    for item in uncertain_data:
                        if item['原始行号'] == term2_info['row_num'] and '相似' in item['具体问题']:
                            already_exists = True
                            break
                    
                    if not already_exists:
                        uncertain_data.append({
                            '原始行号': term2_info['row_num'],
                            source_col: term2_info['source'],
                            target_col: term2_info['target'],
                            '问题大类': category,
                            '具体问题': detail,
                            '冲突组ID': similar_group_id  # 使用相似组ID
                        })
        else:
            print("  未发现高度相似的术语")
    
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
                # 多个质量问题，合并
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
    
    # ========== 所有检测完成，开始保存文件和生成报告 ==========
    print("\n" + "="*60)
    print("正在保存结果...")
    print("="*60)
    
    # 保存清洗后的数据
    if file_ext == '.csv':
        df_cleaned.to_csv(output_clean, index=False, encoding='utf-8-sig')
    else:
        df_cleaned.to_excel(output_clean, index=False)
    print(f"\n✓ 清洗后的术语表已保存: {output_clean}")
    print(f"  保留条目数: {len(df_cleaned)} ({len(df_cleaned)/len(df)*100:.2f}%)")
    
    # 保存问题条目
    if all_removed_data:
        df_removed = pd.DataFrame(all_removed_data)
        
        # 优化1：拆分"具体问题"为多列（问题1、问题2、问题3...）
        max_issues = 0
        split_issues = []
        
        for item in all_removed_data:
            # 按分号或逗号拆分多个问题
            issues = [issue.strip() for issue in item['具体问题'].replace('；', ';').split(';')]
            split_issues.append(issues)
            max_issues = max(max_issues, len(issues))
        
        # 添加拆分后的问题列
        for i in range(max_issues):
            col_name = f'问题{i+1}' if max_issues > 1 else '具体问题'
            df_removed[col_name] = [issues[i] if i < len(issues) else '' for issues in split_issues]
        
        # 删除原来的"具体问题"列（已经拆分了）
        if max_issues > 1:
            df_removed = df_removed.drop(columns=['具体问题'])
        
        # 先重命名'冲突组ID'为'组ID'（统一命名）
        if '冲突组ID' in df_removed.columns:
            df_removed = df_removed.rename(columns={'冲突组ID': '组ID'})
        
        # 按严重程度、问题大类、组ID、原始行号排序
        severity_order = {'严重': 0, '警告': 1, '提示': 2}
        df_removed['_severity_order'] = df_removed['严重程度'].map(severity_order)
        sort_cols = ['_severity_order', '问题大类']
        if '组ID' in df_removed.columns:
            sort_cols.append('组ID')
        sort_cols.append('原始行号')
        df_removed = df_removed.sort_values(sort_cols)
        df_removed = df_removed.drop(columns=['_severity_order'])
        
        # 优化2：组ID只在每组的第一行显示（包括冲突组G和相似组S）
        if '组ID' in df_removed.columns:
            prev_group = None
            clean_group_ids = []
            for group_id in df_removed['组ID']:
                if group_id and group_id != prev_group:
                    # 新的组，显示组ID
                    clean_group_ids.append(group_id)
                    prev_group = group_id
                elif group_id == prev_group:
                    # 同一组的后续行，留空
                    clean_group_ids.append('')
                else:
                    # 没有组ID
                    clean_group_ids.append('')
            df_removed['组ID'] = clean_group_ids
        
        # 重新排列列顺序：严重程度、原始行号、问题大类、组ID（如果有）、原文、译文、问题列
        base_cols = ['严重程度', '原始行号', '问题大类']
        if '组ID' in df_removed.columns:
            base_cols.append('组ID')
        
        # 添加原文和译文列
        term_cols = [source_col, target_col]
        
        # 添加问题列
        issue_cols = [col for col in df_removed.columns if col.startswith('问题') or col == '具体问题']
        
        # 其他列
        other_cols = [col for col in df_removed.columns if col not in base_cols + term_cols + issue_cols]
        
        # 重新排列
        cols = base_cols + term_cols + issue_cols + other_cols
        df_removed = df_removed[cols]
        
        if file_ext == '.csv':
            df_removed.to_csv(output_removed, index=False, encoding='utf-8-sig')
        else:
            df_removed.to_excel(output_removed, index=False)
        
        print(f"✓ 问题条目已保存: {output_removed}")
        print(f"  问题条目数: {len(df_removed)} ({len(df_removed)/len(df)*100:.2f}%)")
    
    # 计算统计信息
    if len(df_cleaned) > 0:
        stats = {
            '原文字符数': df_cleaned[source_col].astype(str).str.len(),
            '译文字符数': df_cleaned[target_col].astype(str).str.len()
        }
    else:
        # 如果清洗后没有数据，使用占位符
        stats = None
    
    # 生成并打印清洗报告
    print("\n" + "="*60)
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
"""
    
    if stats is not None:
        report += f"""
术语长度统计（清洗后）:
  - 原文平均长度: {stats['原文字符数'].mean():.1f} 字符
  - 原文最短: {stats['原文字符数'].min()} 字符
  - 原文最长: {stats['原文字符数'].max()} 字符
  - 译文平均长度: {stats['译文字符数'].mean():.1f} 字符
  - 译文最短: {stats['译文字符数'].min()} 字符
  - 译文最长: {stats['译文字符数'].max()} 字符
"""
    else:
        report += """
术语长度统计（清洗后）:
  - 无数据（所有条目都存在问题）
"""
    
    report += """
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
    
    report += f"""
输出文件:
  - 清洗后术语表: {output_clean}
  - 问题条目表: {output_removed}"""
    
    report += """

说明:
  - cleaned 文件：完全干净的术语表，可直接使用
  - removed 文件：所有问题条目，按严重程度排序
    * 严重：必须处理（空值、重复、语言错误、数字不一致）
    * 警告：建议检查（翻译冲突、长度异常、相似术语）
    * 提示：可选检查（格式问题等）
    * 相似术语问题（如果启用）已包含在此文件中"""
    
    report += """
=======================================================
"""
    
    print(report)
    
    # 保存报告
    report_file = f"{os.path.splitext(input_file)[0]}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 报告已保存: {report_file}\n")
    
    return {
        'success': True,
        'cleaned_count': len(df_cleaned),
        'removed_count': total_removed
    }


def batch_process_directory(directory=".", config=None):
    """
    批量处理目录下的所有Excel和CSV文件
    
    参数:
        directory: 目标目录路径（默认为当前目录）
        config: 清洗配置对象（CleanConfig实例）
    """
    if config is None:
        config = CleanConfig()
    
    # 查找所有Excel和CSV文件（递归查找子目录）
    import glob
    files = []
    for pattern in ['*.xlsx', '*.xls', '*.csv']:
        files.extend(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
    
    # 过滤掉已经处理过的文件（_cleaned, _removed, _uncertain, _similar 结尾）
    files = [f for f in files if not any(suffix in f for suffix in ['_cleaned', '_removed', '_uncertain', '_similar', '_report'])]
    
    if not files:
        print(f"目录 {directory} 中没有找到需要处理的Excel或CSV文件")
        return
    
    print(f"找到 {len(files)} 个文件待处理:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    
    print("\n开始批量处理...")
    print("=" * 60)
    
    results = []
    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] 正在处理: {os.path.basename(file)}")
        print("-" * 60)
        
        try:
            result = clean_terminology_table(file, config=config)
            results.append({
                'file': os.path.basename(file),
                'success': True,
                'cleaned_count': result.get('cleaned_count', 0),
                'removed_count': result.get('removed_count', 0)
            })
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            results.append({
                'file': os.path.basename(file),
                'success': False,
                'error': str(e)
            })
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("批量处理完成！")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.get('success', False))
    fail_count = len(results) - success_count
    
    print(f"\n成功: {success_count}/{len(results)}")
    if fail_count > 0:
        print(f"失败: {fail_count}/{len(results)}")
    
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

