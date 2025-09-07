import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 0

# 模拟三种特征融合策略的性能数据
np.random.seed(42)

# 定义三种融合策略
fusion_strategies = ['Concatenation', 'Cross Attention', 'Joint Graph']
colors = ['#FFB6C1', '#98FB98', '#DDA0DD']  # 对应SVG中的颜色

# 模拟训练过程数据
epochs = 50

# 生成模拟的训练和验证损失
def generate_loss_data(strategy_idx):
    base_train_loss = 0.65 - strategy_idx * 0.05
    base_val_loss = 0.70 - strategy_idx * 0.03
    
    train_loss = []
    val_loss = []
    
    for epoch in range(epochs):
        # 训练损失逐渐下降
        train_noise = np.random.normal(0, 0.02)
        train_val = base_train_loss * np.exp(-epoch * 0.08) + 0.35 + train_noise
        train_loss.append(max(0.3, train_val))
        
        # 验证损失
        val_noise = np.random.normal(0, 0.03)
        val_val = base_val_loss * np.exp(-epoch * 0.06) + 0.38 + val_noise
        val_loss.append(max(0.35, val_val))
    
    return train_loss, val_loss

# 生成模拟的验证准确率
def generate_accuracy_data(strategy_idx):
    base_acc = 0.75 + strategy_idx * 0.05
    accuracy = []
    
    for epoch in range(epochs):
        if epoch < 5:
            acc = 0.5 + (base_acc - 0.5) * (epoch / 5) + np.random.normal(0, 0.02)
        else:
            acc = base_acc + np.random.normal(0, 0.015)
        accuracy.append(min(0.95, max(0.5, acc)))
    
    return accuracy

# 生成模拟的混淆矩阵数据
def generate_confusion_matrix(strategy_idx):
    # 基于策略性能生成不同的混淆矩阵
    if strategy_idx == 0:  # Concatenation
        return np.array([[180, 152], [65, 603]])
    elif strategy_idx == 1:  # Cross Attention
        return np.array([[195, 137], [58, 610]])
    else:  # Joint Graph
        return np.array([[210, 122], [45, 623]])

# 生成模拟的ROC曲线数据
def generate_roc_data(strategy_idx):
    n_samples = 1000
    # 不同策略有不同的AUC值
    auc_values = [0.82, 0.87, 0.91]
    
    # 生成模拟的真实标签和预测概率
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # 根据AUC值调整预测概率
    base_prob = np.random.beta(2, 5, n_samples)
    y_scores = base_prob + (auc_values[strategy_idx] - 0.5) * y_true + np.random.normal(0, 0.1, n_samples)
    y_scores = np.clip(y_scores, 0, 1)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

# 创建可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feature Fusion Strategies Performance Comparison\n特征融合策略性能比较', fontsize=16, fontweight='bold')

# 1. 训练和验证损失对比
ax1 = axes[0, 0]
for i, strategy in enumerate(fusion_strategies):
    train_loss, val_loss = generate_loss_data(i)
    ax1.plot(range(epochs), train_loss, label=f'{strategy} - Training', color=colors[i], linestyle='-', alpha=0.8, linewidth=2)
    ax1.plot(range(epochs), val_loss, label=f'{strategy} - Validation', color=colors[i], linestyle='--', alpha=0.8, linewidth=2)

ax1.set_title('Training and Validation Loss\n训练和验证损失', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=10)
ax1.set_ylabel('Loss', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, epochs-1)
ax1.set_ylim(0.3, 0.7)

# 2. 验证准确率对比
ax2 = axes[0, 1]
for i, strategy in enumerate(fusion_strategies):
    accuracy = generate_accuracy_data(i)
    ax2.plot(range(epochs), accuracy, label=strategy, color=colors[i], linewidth=2, marker='o', markersize=2, alpha=0.8)

ax2.set_title('Validation Accuracy Comparison\n验证准确率比较', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Accuracy', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.5, 1.0)
ax2.set_xlim(0, epochs-1)

# 3. ROC曲线对比
ax3 = axes[0, 2]
for i, strategy in enumerate(fusion_strategies):
    fpr, tpr, roc_auc = generate_roc_data(i)
    ax3.plot(fpr, tpr, color=colors[i], linewidth=3, 
             label=f'{strategy} (AUC = {roc_auc:.3f})', alpha=0.8)

ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
ax3.set_title('ROC Curves Comparison\nROC曲线比较', fontsize=12, fontweight='bold')
ax3.set_xlabel('False Positive Rate', fontsize=10)
ax3.set_ylabel('True Positive Rate', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# 4. 性能指标对比柱状图
ax4 = axes[1, 0]
metrics = ['Accuracy\n准确率', 'Precision\n精确率', 'Recall\n召回率', 'F1-Score']
concat_scores = [0.82, 0.80, 0.85, 0.82]
cross_att_scores = [0.87, 0.85, 0.88, 0.86]
joint_graph_scores = [0.91, 0.89, 0.92, 0.90]

x = np.arange(len(metrics))
width = 0.25

ax4.bar(x - width, concat_scores, width, label='Concatenation', color=colors[0], alpha=0.8)
ax4.bar(x, cross_att_scores, width, label='Cross Attention', color=colors[1], alpha=0.8)
ax4.bar(x + width, joint_graph_scores, width, label='Joint Graph', color=colors[2], alpha=0.8)

ax4.set_title('Performance Metrics Comparison\n性能指标对比')
ax4.set_ylabel('Score')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.set_ylim(0, 1)
ax4.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (concat, cross, joint) in enumerate(zip(concat_scores, cross_att_scores, joint_graph_scores)):
    ax4.text(i - width, concat + 0.01, f'{concat:.2f}', ha='center', va='bottom', fontsize=8)
    ax4.text(i, cross + 0.01, f'{cross:.2f}', ha='center', va='bottom', fontsize=8)
    ax4.text(i + width, joint + 0.01, f'{joint:.2f}', ha='center', va='bottom', fontsize=8)

# 5. 混淆矩阵热力图（选择最佳策略：Joint Graph）
ax5 = axes[1, 1]
cm = generate_confusion_matrix(2)  # Joint Graph的混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, 
            xticklabels=['0', '1'],
            yticklabels=['0', '1'])
ax5.set_title('Confusion Matrix (Joint Graph)\n混淆矩阵（联合图方法）')

# 6. 特征融合策略架构对比
ax6 = axes[1, 2]
# 创建一个简化的架构对比图
strategy_names = ['Concatenation\n串联', 'Cross Attention\n交叉注意力', 'Joint Graph\n联合图']
complexity_scores = [3, 7, 9]  # 复杂度评分
performance_scores = [6, 8, 9]  # 性能评分

x_pos = np.arange(len(strategy_names))
width = 0.35

rects1 = ax6.bar(x_pos - width/2, complexity_scores, width, label='Complexity\n复杂度', color='lightcoral', alpha=0.7)
rects2 = ax6.bar(x_pos + width/2, performance_scores, width, label='Performance\n性能', color='lightblue', alpha=0.7)

ax6.set_title('Strategy Complexity vs Performance\n策略复杂度与性能对比')
ax6.set_ylabel('Score (1-10)')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(strategy_names)
ax6.legend()
ax6.set_ylim(0, 10)
ax6.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for rect in rects1:
    height = rect.get_height()
    ax6.text(rect.get_x() + rect.get_width()/2., height + 0.1,
             f'{int(height)}', ha='center', va='bottom')

for rect in rects2:
    height = rect.get_height()
    ax6.text(rect.get_x() + rect.get_width()/2., height + 0.1,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('feature_fusion_strategies_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()

# 打印总结报告
print("\n" + "="*60)
print("特征融合策略性能分析报告")
print("Feature Fusion Strategies Performance Analysis Report")
print("="*60)

print("\n1. 策略概述 (Strategy Overview):")
print("   • Concatenation (串联): 简单直接的特征拼接")
print("   • Cross Attention (交叉注意力): 动态权重分配")
print("   • Joint Graph (联合图): 结构化交互建模")

print("\n2. 性能排名 (Performance Ranking):")
print("   1st: Joint Graph - 最高准确率 (91%)")
print("   2nd: Cross Attention - 平衡性能 (87%)")
print("   3rd: Concatenation - 基础性能 (82%)")

print("\n3. 建议 (Recommendations):")
print("   • 对于高精度要求: 推荐 Joint Graph")
print("   • 对于计算效率: 推荐 Cross Attention")
print("   • 对于简单场景: 推荐 Concatenation")

print("\n可视化图表已保存为: feature_fusion_strategies_comparison.png")
print("Visualization saved as: feature_fusion_strategies_comparison.png")