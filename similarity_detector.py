"""
相似度检测模块
使用混合方法：字符串相似度 + 语义相似度
"""

import pandas as pd
from logger import get_logger

logger = get_logger()

# 相似度检测相关导入（可选）
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
    logger.debug("rapidfuzz 可用")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz 不可用，相似度检测功能将受限")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.debug("sentence_transformers 可用")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers 不可用，相似度检测功能将受限")


class SimilarityDetector:
    """
    相似术语检测器
    使用混合方法：字符串相似度 + 语义相似度
    """
    def __init__(self, config=None):
        from config import CleanConfig
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
    
    def _quick_filter(self, term1, term2):
        """
        快速预筛选：明显不相似的术语对直接返回False
        """
        len1, len2 = len(term1), len(term2)
        
        # 长度差异超过50%，直接跳过
        if max(len1, len2) > 0:
            length_ratio = min(len1, len2) / max(len1, len2)
            if length_ratio < 0.5:
                return False
        
        # 都是英文且首字母不同，降低优先级（但不完全排除）
        if term1 and term2:
            if term1[0].isalpha() and term2[0].isalpha():
                if term1[0].lower() != term2[0].lower():
                    # 首字母不同但长度相似，可能是拼写错误，保留
                    if abs(len1 - len2) <= 2:
                        return True
                    # 否则跳过
                    return False
        
        return True
    
    def detect_similar_terms(self, terms, column_name='术语'):
        """
        两阶段检测相似术语（优化版）
        
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
        
        # 安全检查：对于超大数据集，给出警告
        total_comparisons = len(unique_terms) * (len(unique_terms) - 1) // 2
        if total_comparisons > 1000000:  # 超过100万对
            print(f"  警告：数据量较大（约 {total_comparisons/1000000:.1f}M 对比较），处理时间可能较长")
            print(f"  建议：可以提高相似度阈值（当前85%）来加快速度")
        
        similar_pairs = []
        
        # 第一阶段：字符相似度快速筛选（带预过滤）
        print("  阶段1: 字符相似度筛选（含快速预筛选）...")
        candidates = []
        
        # 优化：使用rapidfuzz批量处理（如果可用）
        if RAPIDFUZZ_AVAILABLE:
            from rapidfuzz import process, fuzz as rf_fuzz
            
            total_terms = len(unique_terms)
            skipped_count = 0
            
            for i, term1 in enumerate(unique_terms):
                # 显示进度（仅每5%更新一次，减少IO开销）
                if i == 0 or i % max(1, total_terms // 20) == 0 or i == total_terms - 1:
                    print(f"    处理进度: {i}/{total_terms} ({i/total_terms*100:.1f}%)，已跳过 {skipped_count} 对")
                
                # 批量计算当前term1与所有后续术语的相似度
                remaining_terms = unique_terms[i+1:]
                if remaining_terms:
                    # 快速预筛选：过滤掉明显不相似的
                    filtered_terms = []
                    filtered_indices = []
                    for j, term2 in enumerate(remaining_terms):
                        if self._quick_filter(term1, term2):
                            filtered_terms.append(term2)
                            filtered_indices.append(j)
                        else:
                            skipped_count += 1
                    
                    # 只对通过预筛选的术语进行详细比较
                    if filtered_terms:
                        # 使用rapidfuzz的批量API
                        results = process.cdist([term1], filtered_terms, 
                                              scorer=rf_fuzz.ratio, 
                                              dtype=None)
                        
                        # 筛选相似度 > 50% 的
                        for j, sim_score in enumerate(results[0]):
                            char_sim = sim_score / 100.0
                            if char_sim > 0.50:
                                candidates.append((term1, filtered_terms[j], char_sim))
            
            # 确保显示100%进度
            print(f"    处理进度: {total_terms}/{total_terms} (100.0%)，已跳过 {skipped_count} 对")
        else:
            # 备用方法：逐个计算（带预筛选）
            skipped_count = 0
            for i, term1 in enumerate(unique_terms):
                for term2 in unique_terms[i+1:]:
                    # 快速预筛选
                    if not self._quick_filter(term1, term2):
                        skipped_count += 1
                        continue
                    
                    char_sim = self.char_similarity(term1, term2)
                    
                    # 字符相似度 > 50% 的进入第二阶段
                    if char_sim > 0.50:
                        candidates.append((term1, term2, char_sim))
            
            print(f"    已跳过 {skipped_count} 对明显不相似的术语")
        
        print(f"  找到 {len(candidates)} 对候选相似术语")
        
        if not candidates:
            return []
        
        # 第二阶段：语义相似度精确计算（批量优化）
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("  阶段2: 语义相似度计算（批量模式）...")
            self.load_semantic_model()
            
            # 导入torch（sentence_transformers已经包含）
            try:
                import torch
            except ImportError:
                print("    错误：需要PyTorch支持")
                return []
            
            # 优化：批量编码所有涉及的唯一术语
            unique_candidate_terms = set()
            for term1, term2, _ in candidates:
                unique_candidate_terms.add(term1)
                unique_candidate_terms.add(term2)
            
            unique_candidate_terms = list(unique_candidate_terms)
            print(f"    正在批量编码 {len(unique_candidate_terms)} 个术语...")
            
            # 分批编码（避免编码阶段内存问题）
            encode_batch_size = 500  # 每批编码500个术语
            all_embeddings = []
            
            if len(unique_candidate_terms) > encode_batch_size:
                print(f"    分批编码（每批{encode_batch_size}个术语）...")
                for i in range(0, len(unique_candidate_terms), encode_batch_size):
                    batch_terms = unique_candidate_terms[i:i+encode_batch_size]
                    batch_emb = self.semantic_model.encode(batch_terms, 
                                                          convert_to_tensor=True,
                                                          show_progress_bar=False,
                                                          batch_size=32)
                    all_embeddings.append(batch_emb)
                    
                    if (i // encode_batch_size + 1) % 5 == 0:
                        print(f"      已编码 {min(i+encode_batch_size, len(unique_candidate_terms))}/{len(unique_candidate_terms)} 个术语")
                
                # 合并所有批次的嵌入
                embeddings = torch.cat(all_embeddings, dim=0)
                del all_embeddings
            else:
                # 数量不多，直接编码
                embeddings = self.semantic_model.encode(unique_candidate_terms, 
                                                        convert_to_tensor=True,
                                                        show_progress_bar=False,
                                                        batch_size=32)
            
            # 创建术语到索引和向量的映射
            term_to_idx = {term: i for i, term in enumerate(unique_candidate_terms)}
            
            print(f"    正在计算 {len(candidates)} 对术语的相似度...")
            
            # 向量化计算：收集所有需要计算的索引对
            idx_pairs = []
            candidate_map = {}  # 记录每个索引对对应的候选信息
            
            for idx, (term1, term2, char_sim) in enumerate(candidates):
                i1 = term_to_idx[term1]
                i2 = term_to_idx[term2]
                idx_pairs.append((i1, i2))
                candidate_map[idx] = (term1, term2, char_sim)
            
            # 批量计算余弦相似度（分批处理，避免内存爆炸）
            if idx_pairs:
                # 分批处理：每批最多1000对，避免内存问题
                batch_size = 1000
                total_pairs = len(idx_pairs)
                
                for batch_start in range(0, total_pairs, batch_size):
                    batch_end = min(batch_start + batch_size, total_pairs)
                    batch_pairs = idx_pairs[batch_start:batch_end]
                    
                    # 显示进度（每10批或每20%显示一次）
                    current_batch = batch_start // batch_size + 1
                    total_batches = (total_pairs + batch_size - 1) // batch_size
                    if total_pairs > batch_size and (current_batch % 10 == 1 or current_batch % max(1, total_batches // 5) == 0):
                        print(f"    处理批次: {current_batch}/{total_batches} ({current_batch/total_batches*100:.0f}%)")
                    
                    # 提取当前批次的索引
                    indices1 = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long)
                    indices2 = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long)
                    
                    # 获取对应的嵌入向量
                    emb1_batch = embeddings[indices1]
                    emb2_batch = embeddings[indices2]
                    
                    # 批量计算余弦相似度
                    semantic_sims = util.cos_sim(emb1_batch, emb2_batch).diagonal().cpu().numpy()
                    
                    # 处理当前批次的结果
                    for local_idx, semantic_sim in enumerate(semantic_sims):
                        global_idx = batch_start + local_idx
                        term1, term2, char_sim = candidate_map[global_idx]
                        
                        # 混合相似度计算
                        final_sim = (self.config.char_sim_weight * char_sim + 
                                    self.config.semantic_sim_weight * float(semantic_sim))
                        
                        # 最终相似度超过阈值
                        if final_sim >= self.config.similarity_threshold:
                            similar_pairs.append({
                                '列名': column_name,
                                '术语1': term1,
                                '术语2': term2,
                                '相似度': f"{final_sim*100:.1f}%",
                                '字符相似度': f"{char_sim*100:.1f}%",
                                '语义相似度': f"{float(semantic_sim)*100:.1f}%",
                                '检测方法': '混合相似度'
                            })
                    
                    # 清理内存
                    del emb1_batch, emb2_batch, semantic_sims, indices1, indices2
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            print(f"    完成！")
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

