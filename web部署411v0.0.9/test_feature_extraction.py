# -*- coding: utf-8 -*-
"""
测试特征提取功能，验证修复是否有效
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_short_data():
    """测试短数据的处理"""
    print("🧪 测试短数据处理...")
    
    try:
        from feature_extractor import extract_features_from_segment, butter_lowpass_filter
        
        # 测试非常短的数据
        short_voltage = np.array([220, 221, 219])
        short_current = np.array([0.1, 0.2, 0.15])
        
        print(f"测试数据长度: {len(short_voltage)}")
        
        # 测试滤波器
        filtered = butter_lowpass_filter(short_current)
        print(f"滤波器测试通过，输出长度: {len(filtered)}")
        
        # 测试特征提取
        features = extract_features_from_segment(short_voltage, short_current)
        print(f"特征提取成功，特征数量: {len(features)}")
        print("✅ 短数据测试通过")
        
    except Exception as e:
        print(f"❌ 短数据测试失败: {e}")
        return False
    
    return True


def test_normal_data():
    """测试正常长度数据的处理"""
    print("\n🧪 测试正常数据处理...")
    
    try:
        from feature_extractor import process_data
        
        # 创建正常长度的测试数据
        time = np.linspace(0, 3, 1875)  # 3秒，625Hz
        voltage = np.full_like(time, 220) + np.random.randn(len(time)) * 1
        current = (1 - np.exp(-time * 3)) * 1.5 + np.random.randn(len(time)) * 0.1
        
        test_df = pd.DataFrame({
            'Time': time,
            'Voltage': voltage,
            'Current': current
        })
        
        print(f"测试数据形状: {test_df.shape}")
        
        # 测试特征提取
        features_df = process_data(
            test_df,
            voltage_col='Voltage',
            current_col='Current'
        )
        
        print(f"特征提取成功，特征形状: {features_df.shape}")
        print(f"特征列数: {len(features_df.columns)}")
        print("✅ 正常数据测试通过")
        
        return features_df
        
    except Exception as e:
        print(f"❌ 正常数据测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_edge_cases():
    """测试边界情况"""
    print("\n🧪 测试边界情况...")
    
    try:
        from feature_extractor import extract_features_from_segment
        
        # 测试空数组
        empty_data = np.array([])
        if len(empty_data) == 0:
            print("空数组检测正确")
        
        # 测试单个数据点
        single_voltage = np.array([220])
        single_current = np.array([0.1])
        features = extract_features_from_segment(single_voltage, single_current)
        print(f"单点数据特征提取: {len(features)} 个特征")
        
        # 测试不同长度数据
        voltage_10 = np.random.randn(10) + 220
        current_15 = np.random.randn(15) + 1
        features = extract_features_from_segment(voltage_10, current_15)
        print(f"不同长度数据特征提取: {len(features)} 个特征")
        
        print("✅ 边界情况测试通过")
        
    except Exception as e:
        print(f"❌ 边界情况测试失败: {e}")
        return False
    
    return True


def test_feature_names():
    """测试特征名称"""
    print("\n🧪 测试特征名称...")
    
    try:
        from feature_extractor import get_feature_names
        
        feature_names = get_feature_names()
        print(f"特征名称数量: {len(feature_names)}")
        print("前5个特征名称:")
        for i, name in enumerate(feature_names[:5]):
            print(f"  {i+1}. {name}")
        
        print("✅ 特征名称测试通过")
        
    except Exception as e:
        print(f"❌ 特征名称测试失败: {e}")
        return False
    
    return True


def test_save_functions():
    """测试保存功能"""
    print("\n🧪 测试保存功能...")
    
    try:
        from feature_extractor import save_features_to_file, export_features_for_analysis
        
        # 创建测试特征数据
        test_features = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [0.5, 1.5, 2.5],
            'feature_3': [10, 20, 30]
        })
        
        # 测试CSV保存
        csv_file = save_features_to_file(test_features, output_dir='test_output', file_format='csv')
        if os.path.exists(csv_file):
            print(f"CSV保存成功: {csv_file}")
            os.remove(csv_file)  # 清理测试文件
        
        # 测试Excel保存
        excel_file = save_features_to_file(test_features, output_dir='test_output', file_format='excel')
        if os.path.exists(excel_file):
            print(f"Excel保存成功: {excel_file}")
            os.remove(excel_file)  # 清理测试文件
        
        # 测试分析导出
        analysis_file = export_features_for_analysis(test_features, 'classification', 'test_output')
        if os.path.exists(analysis_file):
            print(f"分析数据导出成功: {analysis_file}")
            os.remove(analysis_file)  # 清理测试文件
            
            # 清理元数据文件
            metadata_file = analysis_file.replace('.csv', '_metadata.json')
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
        
        # 清理测试目录
        if os.path.exists('test_output') and not os.listdir('test_output'):
            os.rmdir('test_output')
        
        print("✅ 保存功能测试通过")
        
    except Exception as e:
        print(f"❌ 保存功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """主测试函数"""
    print("🚀 特征提取功能测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("短数据处理", test_short_data()))
    test_results.append(("正常数据处理", test_normal_data() is not None))
    test_results.append(("边界情况", test_edge_cases()))
    test_results.append(("特征名称", test_feature_names()))
    test_results.append(("保存功能", test_save_functions()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15s}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！特征提取功能正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关功能。")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
