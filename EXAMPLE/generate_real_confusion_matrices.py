import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入必要的类
try:
    from dengue_ns5_inhibitor_prediction import (
        MolecularGraphDataset, 
        ProteinGraphProcessor, 
        DengueNS5InhibitorPredictor,
        load_and_preprocess_data
    )
    print("✅ 成功导入所需模块")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保 dengue_ns5_inhibitor_prediction.py 文件在当前目录中")
    exit(1)

def load_trained_model(model_path, fusion_strategy):
    """加载训练好的模型"""
    try:
        print(f"正在加载模型: {model_path}")
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
            
        # 创建模型实例
        model = DengueNS5InhibitorPredictor(fusion_strategy=fusion_strategy)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  从checkpoint加载模型状态")
        else:
            model.load_state_dict(checkpoint)
            print(f"  直接加载模型状态")
        
        model.eval()
        print(f"✅ 模型 {model_path} 加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 加载模型 {model_path} 失败: {e}")
        return None

def evaluate_model_predictions(model, test_loader, protein_data, strategy_name):
    """评估模型并获取预测结果"""
    print(f"正在评估 {strategy_name} 模型...")
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_idx, (batch_mol, batch_labels) in enumerate(test_loader):
            try:
                outputs = model(batch_mol, protein_data)
                predictions = (outputs > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(batch_labels.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  处理批次 {batch_idx+1}/{len(test_loader)}")
                    
            except Exception as e:
                print(f"  批次 {batch_idx} 处理失败: {e}")
                continue
    
    print(f"✅ {strategy_name} 模型评估完成，共处理 {len(all_predictions)} 个样本")
    return np.array(all_true_labels), np.array(all_predictions)

def generate_real_confusion_matrices():
    """生成基于真实模型的混淆矩阵对比图"""
    print("=== 生成基于真实模型的混淆矩阵对比 ===")
    
    try:
        # 1. 加载数据
        print("\n1. 加载数据...")
        smiles_list, labels = load_and_preprocess_data()
        print(f"✅ 数据加载完成，共 {len(smiles_list)} 个化合物")
        print(f"   活性化合物: {sum(labels)} 个")
        print(f"   非活性化合物: {len(labels) - sum(labels)} 个")
        
        # 2. 处理蛋白质结构
        print("\n2. 处理蛋白质结构...")
        protein_processor = ProteinGraphProcessor('Dengue virus 3序列.pdb')
        protein_data = protein_processor.protein_graph
        print("✅ 蛋白质结构处理完成")
        
        # 3. 创建数据集
        print("\n3. 创建数据集...")
        dataset = MolecularGraphDataset(smiles_list, labels)
        print(f"✅ 数据集创建完成，共 {len(dataset)} 个样本")
        
        # 4. 数据分割（使用相同的随机种子确保一致性）
        print("\n4. 数据分割...")
        train_indices, test_indices = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42, stratify=labels
        )
        print(f"✅ 训练集: {len(train_indices)} 个样本，测试集: {len(test_indices)} 个样本")
        
        # 创建测试数据加载器
        def collate_fn(batch):
            mol_graphs, labels = zip(*batch)
            return Batch.from_data_list(mol_graphs), torch.tensor(labels, dtype=torch.float)
        
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        print(f"✅ 测试数据加载器创建完成，共 {len(test_loader)} 个批次")
        
        # 5. 模型配置
        model_configs = [
            {'path': 'best_dengue_ns5_model_concat.pth', 'strategy': 'concat', 'name': 'Concat'},
            {'path': 'best_dengue_ns5_model_cross_attention.pth', 'strategy': 'cross_attention', 'name': 'Cross Attention'},
            {'path': 'best_dengue_ns5_model_joint_graph.pth', 'strategy': 'joint_graph', 'name': 'Joint Graph'}
        ]
        
        # 6. 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('三种融合策略的真实混淆矩阵对比\nReal Confusion Matrix Comparison of Three Fusion Strategies', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        results_summary = []
        successful_models = 0
        
        for idx, config in enumerate(model_configs):
            print(f"\n5.{idx+1} 处理 {config['name']} 模型...")
            
            # 加载模型
            model = load_trained_model(config['path'], config['strategy'])
            if model is None:
                print(f"⚠️ 跳过 {config['name']} 模型")
                # 创建空的混淆矩阵显示
                ax = axes[idx]
                ax.text(0.5, 0.5, f'{config["name"]}\n模型加载失败', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{config["name"]} (模型未找到)', fontweight='bold', fontsize=12)
                ax.set_xlabel('Predicted Label', fontweight='bold')
                ax.set_ylabel('True Label', fontweight='bold')
                continue
            
            # 获取预测结果
            y_true, y_pred = evaluate_model_predictions(model, test_loader, protein_data, config['name'])
            
            if len(y_true) == 0:
                print(f"❌ {config['name']} 模型预测失败")
                ax = axes[idx]
                ax.text(0.5, 0.5, f'{config["name"]}\n预测失败', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{config["name"]} (预测失败)', fontweight='bold', fontsize=12)
                continue
            
            successful_models += 1
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            print(f"✅ {config['name']} 混淆矩阵:\n{cm}")
            
            # 计算性能指标
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results_summary.append({
                'Model': config['name'],
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            })
            
            # 绘制混淆矩阵
            ax = axes[idx]
            
            # 使用seaborn绘制热图
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['0', '1'], 
                       yticklabels=['0', '1'],
                       ax=ax, cbar_kws={'shrink': 0.8})
            
            ax.set_title(f'{config["name"]}\nAcc: {accuracy:.3f}, F1: {f1:.3f}', 
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_ylabel('True Label', fontweight='bold')
            
            # 添加性能文本
            textstr = f'TP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
            
            print(f"   准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
        
        plt.tight_layout()
        plt.savefig('real_confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ 真实混淆矩阵对比图已保存为 'real_confusion_matrices_comparison.png'")
        plt.close()
        
        # 7. 打印详细结果
        if results_summary:
            print("\n=== 真实模型性能对比总结 ===")
            print("-" * 90)
            print(f"{'模型':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<10}")
            print("-" * 90)
            
            for result in results_summary:
                sample_count = result['TP'] + result['TN'] + result['FP'] + result['FN']
                print(f"{result['Model']:<15} {result['Accuracy']:<10.4f} {result['Precision']:<10.4f} "
                      f"{result['Recall']:<10.4f} {result['F1-Score']:<10.4f} {sample_count:<10}")
            
            print("-" * 90)
            
            # 找出最佳模型
            best_model = max(results_summary, key=lambda x: x['F1-Score'])
            print(f"\n🏆 最佳模型: {best_model['Model']} (F1分数: {best_model['F1-Score']:.4f})")
            
            # 生成详细分类报告
            print("\n=== 详细分类报告 ===")
            for result in results_summary:
                print(f"\n{result['Model']} 模型:")
                print(f"  准确率: {result['Accuracy']:.4f}")
                print(f"  精确率: {result['Precision']:.4f}")
                print(f"  召回率: {result['Recall']:.4f}")
                print(f"  F1分数: {result['F1-Score']:.4f}")
                print(f"  真阳性: {result['TP']}, 真阴性: {result['TN']}")
                print(f"  假阳性: {result['FP']}, 假阴性: {result['FN']}")
        else:
            print("\n❌ 没有成功加载任何模型")
        
        print(f"\n✅ 成功处理了 {successful_models}/3 个模型")
        return results_summary
        
    except Exception as e:
        print(f"❌ 生成真实混淆矩阵时出错: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    try:
        print("开始生成基于真实模型的混淆矩阵对比...")
        results = generate_real_confusion_matrices()
        
        if results:
            print("\n🎉 真实混淆矩阵对比图生成成功！")
            print("📊 生成的文件: real_confusion_matrices_comparison.png")
        else:
            print("\n⚠️ 真实混淆矩阵生成失败")
            print("请检查模型文件是否存在以及数据是否正确加载")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()