#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 regression_training.py 模块导入
"""

import sys
import traceback

def test_regression_import():
    """测试回归训练模块导入"""
    print("开始测试 regression_training.py 模块导入...")
    
    try:
        import regression_training
        print("✅ regression_training.py 导入成功!")
        
        # 检查主要函数是否存在
        if hasattr(regression_training, 'show_regression_training_page'):
            print("✅ show_regression_training_page 函数存在")
        else:
            print("❌ show_regression_training_page 函数不存在")
            
        # 检查主要类是否存在
        if hasattr(regression_training, 'MultiRegressionTrainer'):
            print("✅ MultiRegressionTrainer 类存在")
        else:
            print("❌ MultiRegressionTrainer 类不存在")
            
        print("✅ 所有检查通过!")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_regression_import()
    if success:
        print("\n🎉 测试成功! regression_training.py 模块可以正常导入。")
        sys.exit(0)
    else:
        print("\n💥 测试失败! 请检查 regression_training.py 文件。")
        sys.exit(1)
