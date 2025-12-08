"""
DeepStressModel 程序入口
"""
import sys
import os
import argparse
import logging
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.utils.logger import setup_logger, set_debug_mode

logger = setup_logger("main")

def check_display():
    """检查显示环境是否可用"""
    display = os.environ.get('DISPLAY')
    if not display:
        logger.warning("DISPLAY 环境变量未设置")
        logger.info("提示：在容器中运行时，请使用以下方法之一：")
        logger.info("  1. 运行 ./setup_display.sh 设置显示环境")
        logger.info("  2. 运行 ./run_gui.sh 自动设置并启动")
        logger.info("  3. 使用 X11 转发：docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY ...")
        return False
    
    # 尝试检查显示是否可用（可选，因为 PyQt6 会自己检查）
    try:
        import subprocess
        result = subprocess.run(
            ['xdpyinfo', '-display', display],
            capture_output=True,
            timeout=2
        )
        if result.returncode != 0:
            logger.warning(f"无法连接到显示服务器 {display}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # xdpyinfo 可能不可用，但不影响 PyQt6 运行
        pass
    
    logger.debug(f"显示环境: DISPLAY={display}")
    return True

def main():
    """程序入口函数"""
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description="DeepStressModel - GPU压力测试工具")
        parser.add_argument("--debug", action="store_true", help="启用调试模式，显示详细日志")
        parser.add_argument("--no-display-check", action="store_true", help="跳过显示环境检查")
        args = parser.parse_args()
        
        # 如果启用调试模式，设置所有日志记录器为DEBUG级别
        if args.debug:
            set_debug_mode(True)
            logger.debug("调试模式已启用")
        
        # 检查显示环境
        if not args.no_display_check:
            check_display()
        
        # 创建应用程序实例
        app = QApplication(sys.argv)
        
        # 创建主窗口
        window = MainWindow()
        window.show()
        
        logger.info("程序启动成功")
        
        # 进入事件循环
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
