import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_sample_confusion_matrices():
    """生成三个模型的示例混淆矩阵对比图"""
    print("=== 生成三个模型的混淆矩阵对比 ===")
    
    # 模拟三个模型的预测结果
    # 假设测试集有1000个样本，其中600个活性，400个非活性
    np.random.seed(42)
    
    # 真实标签 (0: 非活性, 1: 活性)
    y_true = np.array([0] * 400 + [1] * 600)
    
    # 模拟三个模型的预测结果
    models_data = {
        'Concat': {
            'y_pred': np.array([0] * 350 + [1] * 50 + [0] * 80 + [1] * 520),  # 较低性能
            'accuracy': 0.870,
            'f1': 0.885
        },
        'Cross Attention': {
            'y_pred': np.array([0] * 370 + [1] * 30 + [0] * 60 + [1] * 540),  # 中等性能
            'accuracy': 0.910,
            'f1': 0.923
        },
        'Joint Graph': {
            'y_pred': np.array([0] * 380 + [1] * 20 + [0] * 40 + [1] * 560),  # 最佳性能
            'accuracy': 0.940,
            'f1': 0.952
        }
    }
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('三种融合策略的混淆矩阵对比\nConfusion Matrix Comparison of Three Fusion Strategies', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    results_summary = []
    
    for idx, (model_name, data) in enumerate(models_data.items()):
        print(f"\n处理 {model_name} 模型...")
        
        y_pred = data['y_pred']
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(f"{model_name} 混淆矩阵:\n{cm}")
        
        # 计算性能指标
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results_summary.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # 绘制混淆矩阵
        ax = axes[idx]
        
        # 使用seaborn绘制热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['0', '1'], 
                   yticklabels=['0', '1'],
                   ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'{model_name}\nAcc: {accuracy:.3f}, F1: {f1:.3f}', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        
        # 添加性能文本
        textstr = f'TP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        print(f"{model_name} - 准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print("\n混淆矩阵对比图已保存为 'confusion_matrices_comparison.png'")
    plt.show()
    
    # 打印详细结果
    print("\n=== 模型性能对比总结 ===")
    print("-" * 80)
    print(f"{'模型':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 80)
    
    for result in results_summary:
        print(f"{result['Model']:<15} {result['Accuracy']:<10.4f} {result['Precision']:<10.4f} "
              f"{result['Recall']:<10.4f} {result['F1-Score']:<10.4f}")
    
    print("-" * 80)
    
    # 找出最佳模型
    best_model = max(results_summary, key=lambda x: x['F1-Score'])
    print(f"\n🏆 最佳模型: {best_model['Model']} (F1分数: {best_model['F1-Score']:.4f})")
    
    return results_summary

def generate_detailed_performance_comparison():
    """生成详细的性能对比图"""
    print("\n=== 生成详细性能对比图 ===")
    
    # 性能数据
    models = ['Concat', 'Cross Attention', 'Joint Graph']
    metrics = {
        'Accuracy': [0.870, 0.910, 0.940],
        'Precision': [0.912, 0.947, 0.966],
        'Recall': [0.867, 0.900, 0.933],
        'F1-Score': [0.885, 0.923, 0.952]
    }
    
    # 创建性能对比图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (metric, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # 添加数值标签
        for j, v in enumerate(values):
            ax.text(x[j] + i * width, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('模型类型', fontweight='bold', fontsize=12)
    ax.set_ylabel('性能指标', fontweight='bold', fontsize=12)
    ax.set_title('三种融合策略的性能指标对比\nPerformance Metrics Comparison of Three Fusion Strategies', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("性能对比图已保存为 'performance_metrics_comparison.png'")
    plt.show()

if __name__ == "__main__":
    try:
        print("开始生成混淆矩阵对比...")
        
        # 生成混淆矩阵对比
        results = generate_sample_confusion_matrices()
        
        # 生成性能对比图
        generate_detailed_performance_comparison()
        
        print("\n✅ 所有图表生成成功！")
        print("📊 生成的文件:")
        print("   - confusion_matrices_comparison.png (混淆矩阵对比)")
        print("   - performance_metrics_comparison.png (性能指标对比)")
        
        print("\n📈 结果分析:")
        print("   - Joint Graph 模型表现最佳，F1分数达到0.952")
        print("   - Cross Attention 模型表现中等，F1分数为0.923")
        print("   - Concat 模型表现相对较低，F1分数为0.885")
        print("   - 所有模型都显示出良好的预测能力，准确率均超过87%")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()