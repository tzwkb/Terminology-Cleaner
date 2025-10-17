"""
双语术语表清洗工具 v2.1
主入口程序
"""

import sys
import os

from config import CleanConfig
from cleaner import clean_terminology_table, batch_process_directory
from logger import setup_logging, get_logger

logger = get_logger()


def main():
    """
    主函数：处理命令行参数或交互式输入
    """
    
    # 设置日志系统
    try:
        setup_logging(level='INFO')
    except Exception as e:
        print(f"警告：日志系统初始化失败: {e}")
    
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
        print("双语术语表清洗工具 v2.1")
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
            config.similarity_check = input("检测相似术语？[Y/n，默认Y]: ").strip().lower() != 'n'
            
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
        config.similarity_check = input("检测相似术语？[Y/n，默认Y]: ").strip().lower() != 'n'
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
        error_msg = f"文件不存在: {input_path}"
        logger.error(error_msg)
        print(f"错误：{error_msg}")
        return
    
    # 检查文件权限
    if not os.access(input_path, os.R_OK):
        error_msg = f"没有读取权限: {input_path}"
        logger.error(error_msg)
        print(f"错误：{error_msg}")
        return
    
    # 执行清洗
    try:
        logger.info(f"开始清洗文件: {input_path}")
        clean_terminology_table(input_path, output_clean, output_removed, output_uncertain, config)
        logger.info("清洗完成")
        print("\n✓ 清洗完成！")
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        print("\n\n⚠ 操作被用户中断")
        return
    except Exception as e:
        logger.error(f"清洗失败: {str(e)}", exc_info=True)
        print(f"\n✗ 清洗过程中出现错误: {str(e)}")
        print("详细错误信息已记录到日志文件")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

