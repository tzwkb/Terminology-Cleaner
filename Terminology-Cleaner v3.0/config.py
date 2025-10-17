"""
配置管理模块
"""


class CleanConfig:
    """清洗配置类"""
    def __init__(self):
        # 基础配置（推荐开启）
        self.ignore_case = True  # 是否忽略大小写
        self.remove_punctuation = True  # 是否去除首尾标点
        self.quality_check = True  # 是否进行质量检查
        
        # 高级配置（按需开启）
        self.interactive_mode = False  # 交互式确认互为翻译（适合小数据集，大数据集建议关闭）
        self.similarity_check = True  # 相似术语检测（需要额外依赖，处理较慢，按需开启）
        
        # 相似度检测参数
        self.similarity_threshold = 0.90  # 相似度阈值（0-1），值越高越严格
        self.char_sim_weight = 0.4  # 字符相似度权重
        self.semantic_sim_weight = 0.6  # 语义相似度权重

