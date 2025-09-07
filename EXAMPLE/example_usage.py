#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
登革热病毒NS5抑制剂预测模型使用示例

这个脚本展示了如何使用训练好的模型进行预测和分析
"""

import torch
import numpy as np
from dengue_ns5_inhibitor_prediction import (
    load_trained_model, 
    predict_inhibitor_activity,
    compare_fusion_strategies,
    load_and_preprocess_data,
    ProteinGraphProcessor,
    MolecularGraphDataset
)

def example_single_prediction():
    """单个分子预测示例"""
    print("=== 单个分子预测示例 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 加载训练好的模型
        model_path = 'best_dengue_ns5_model_concat.pth'
        model, checkpoint = load_trained_model(model_path, device)
        print(f"成功加载模型: {model_path}")
        
        # 加载蛋白质数据
        protein_processor = ProteinGraphProcessor('Dengue virus 3序列.pdb')
        protein_data = protein_processor.protein_graph.to(device)
        
        # 加载分子数据
        smiles_list, labels = load_and_preprocess_data()
        dataset = MolecularGraphDataset(smiles_list, labels)
        sample_mol = dataset[0][0].to(device)  # 取第一个分子作为示例
        
        # 进行预测
        prediction = predict_inhibitor_activity(model, sample_mol, protein_data, device)
        
        print(f"预测结果: {prediction[0]:.4f}")
        print(f"预测类别: {'抑制剂' if prediction[0] > 0.5 else '非抑制剂'}")
        print(f"置信度: {max(prediction[0], 1-prediction[0]):.4f}")
        
    except Exception as e:
        print(f"预测失败: {e}")
        print("请确保已经运行主训练脚本生成模型文件")

def example_batch_prediction():
    """批量预测示例"""
    print("\n=== 批量预测示例 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 加载模型
        model_path = 'best_dengue_ns5_model_cross_attention.pth'
        model, checkpoint = load_trained_model(model_path, device)
        print(f"成功加载模型: {model_path}")
        
        # 加载数据
        protein_processor = ProteinGraphProcessor('Dengue virus 3序列.pdb')
        protein_data = protein_processor.protein_graph.to(device)
        
        smiles_list, labels = load_and_preprocess_data()
        dataset = MolecularGraphDataset(smiles_list, labels)
        
        # 批量预测前5个分子
        batch_mols = [dataset[i][0].to(device) for i in range(5)]
        predictions = predict_inhibitor_activity(model, batch_mols, protein_data, device)
        
        print("\n批量预测结果:")
        for i, (pred, true_label) in enumerate(zip(predictions, labels[:5])):
            print(f"分子 {i+1}: 预测={pred[0]:.4f}, 真实={true_label:.0f}, "
                  f"预测类别={'抑制剂' if pred[0] > 0.5 else '非抑制剂'}")
        
        # 计算准确率
        pred_labels = (predictions > 0.5).astype(int).flatten()
        true_labels = labels[:5].numpy().astype(int)
        accuracy = np.mean(pred_labels == true_labels)
        print(f"\n批量预测准确率: {accuracy:.4f}")
        
    except Exception as e:
        print(f"批量预测失败: {e}")
        print("请确保已经运行主训练脚本生成模型文件")

def example_strategy_comparison():
    """融合策略比较示例"""
    print("\n=== 融合策略比较示例 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 定义模型路径
    model_paths = {
        'concat': 'dengue_ns5_model_concat_complete.pth',
        'cross_attention': 'dengue_ns5_model_cross_attention_complete.pth',
        'joint_graph': 'dengue_ns5_model_joint_graph_complete.pth'
    }
    
    try:
        # 加载测试数据
        protein_processor = ProteinGraphProcessor('Dengue virus 3序列.pdb')
        protein_data = protein_processor.protein_graph.to(device)
        
        smiles_list, labels = load_and_preprocess_data()
        dataset = MolecularGraphDataset(smiles_list, labels)
        test_mols = [dataset[i][0].to(device) for i in range(20)]  # 使用前20个分子作为测试
        
        # 比较不同策略
        results = compare_fusion_strategies(model_paths, test_mols, protein_data, device)
        
        if results:
            print("\n性能排序 (按ROC-AUC):")
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1].get('roc_auc', 0), 
                                  reverse=True)
            
            for i, (strategy, metrics) in enumerate(sorted_results, 1):
                roc_auc = metrics.get('roc_auc', 0)
                accuracy = metrics.get('accuracy', 0)
                f1_score = metrics.get('f1_score', 0)
                print(f"{i}. {strategy}: ROC-AUC={roc_auc:.4f}, "
                      f"Accuracy={accuracy:.4f}, F1={f1_score:.4f}")
        
    except Exception as e:
        print(f"策略比较失败: {e}")
        print("请确保已经运行主训练脚本生成所有模型文件")

def example_model_analysis():
    """模型分析示例"""
    print("\n=== 模型分析示例 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 加载模型
        model_path = 'best_dengue_ns5_model_concat.pth'
        model, checkpoint = load_trained_model(model_path, device)
        
        # 打印模型信息
        print(f"模型配置: {checkpoint.get('model_config', {})}")
        print(f"融合策略: {checkpoint.get('fusion_strategy', 'unknown')}")
        
        # 打印测试指标
        test_metrics = checkpoint.get('test_metrics', {})
        if test_metrics:
            print("\n测试集性能:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # 模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型参数统计:")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数数: {trainable_params:,}")
        
        # 打印模型结构
        print(f"\n模型结构:")
        print(model)
        
    except Exception as e:
        print(f"模型分析失败: {e}")
        print("请确保已经运行主训练脚本生成模型文件")

def main():
    """主函数"""
    print("登革热病毒NS5抑制剂预测模型使用示例")
    print("=" * 50)
    
    # 运行各种示例
    example_single_prediction()
    example_batch_prediction()
    example_strategy_comparison()
    example_model_analysis()
    
    print("\n=== 使用提示 ===")
    print("1. 确保已经运行 dengue_ns5_inhibitor_prediction.py 训练模型")
    print("2. 模型文件应该在当前目录下")
    print("3. 可以修改此脚本来适应你的具体需求")
    print("4. 对于新的分子数据，需要先转换为相同的图表示格式")
    
if __name__ == "__main__":
    main()