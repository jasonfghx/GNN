import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TopHitRecommendationSystem:
    """Top-Hit化合物推荐系统"""
    
    def __init__(self, model, protein_data, device='cpu'):
        self.model = model
        self.protein_data = protein_data
        self.device = device
        self.model.eval()
        
    def predict_compounds(self, smiles_list, batch_size=32):
        """批量预测化合物活性"""
        from dengue_ns5_inhibitor_prediction import MolecularGraphDataset
        
        # 创建临时标签（实际预测时不需要真实标签）
        dummy_labels = [0] * len(smiles_list)
        dataset = MolecularGraphDataset(smiles_list, dummy_labels)
        
        predictions = []
        confidences = []
        
        # 批量处理
        for i in range(0, len(dataset), batch_size):
            batch_data = []
            batch_smiles = []
            
            for j in range(i, min(i + batch_size, len(dataset))):
                mol_data, _ = dataset[j]
                if mol_data is not None:
                    batch_data.append(mol_data)
                    batch_smiles.append(smiles_list[j])
            
            if batch_data:
                # 创建批次
                batch = Batch.from_data_list(batch_data).to(self.device)
                # 确保protein_data在正确的设备上
                protein_data_device = self.protein_data.clone().to(self.device) if hasattr(self.protein_data, 'clone') else self.protein_data.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(batch, protein_data_device)
                    batch_predictions = outputs.cpu().numpy()
                    
                    for pred in batch_predictions:
                        predictions.append(pred)
                        # 计算置信度（距离0.5的距离）
                        confidence = abs(pred - 0.5) * 2
                        confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def calculate_drug_properties(self, smiles_list):
        """计算药物相关性质"""
        properties = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    properties.append({
                        'MW': np.nan, 'LogP': np.nan, 'HBD': np.nan, 'HBA': np.nan,
                        'TPSA': np.nan, 'RotBonds': np.nan, 'Lipinski_Violations': np.nan
                    })
                    continue
                
                # 计算分子性质
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                tpsa = Descriptors.TPSA(mol)
                rot_bonds = Descriptors.NumRotatableBonds(mol)
                
                # Lipinski五规则违反数
                violations = 0
                if mw > 500: violations += 1
                if logp > 5: violations += 1
                if hbd > 5: violations += 1
                if hba > 10: violations += 1
                
                properties.append({
                    'MW': mw,
                    'LogP': logp,
                    'HBD': hbd,
                    'HBA': hba,
                    'TPSA': tpsa,
                    'RotBonds': rot_bonds,
                    'Lipinski_Violations': violations
                })
                
            except Exception as e:
                properties.append({
                    'MW': np.nan, 'LogP': np.nan, 'HBD': np.nan, 'HBA': np.nan,
                    'TPSA': np.nan, 'RotBonds': np.nan, 'Lipinski_Violations': np.nan
                })
        
        return pd.DataFrame(properties)
    
    def rank_compounds(self, smiles_list, weights=None):
        """综合排序化合物"""
        if weights is None:
            weights = {
                'activity_score': 0.5,
                'confidence': 0.2,
                'drug_likeness': 0.2,
                'diversity': 0.1
            }
        
        print(f"正在预测 {len(smiles_list)} 个化合物的活性...")
        predictions, confidences = self.predict_compounds(smiles_list)
        
        print("正在计算药物性质...")
        drug_props = self.calculate_drug_properties(smiles_list)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'SMILES': smiles_list,
            'Activity_Score': predictions,
            'Confidence': confidences
        })
        
        # 合并药物性质
        results = pd.concat([results, drug_props], axis=1)
        
        # 计算药物相似性得分（基于Lipinski规则）
        results['Drug_Likeness_Score'] = self._calculate_drug_likeness_score(results)
        
        # 计算多样性得分（基于分子指纹相似性）
        results['Diversity_Score'] = self._calculate_diversity_score(smiles_list)
        
        # 综合评分
        results['Composite_Score'] = (
            weights['activity_score'] * results['Activity_Score'] +
            weights['confidence'] * results['Confidence'] +
            weights['drug_likeness'] * results['Drug_Likeness_Score'] +
            weights['diversity'] * results['Diversity_Score']
        )
        
        # 按综合得分排序
        results = results.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        results['Rank'] = range(1, len(results) + 1)
        
        return results
    
    def _calculate_drug_likeness_score(self, df):
        """计算药物相似性得分"""
        scores = []
        for _, row in df.iterrows():
            score = 1.0
            
            # Lipinski规则惩罚
            if not pd.isna(row['Lipinski_Violations']):
                score -= row['Lipinski_Violations'] * 0.2
            
            # TPSA优化范围
            if not pd.isna(row['TPSA']):
                if 20 <= row['TPSA'] <= 130:
                    score += 0.1
                else:
                    score -= 0.1
            
            # 旋转键数量
            if not pd.isna(row['RotBonds']):
                if row['RotBonds'] <= 10:
                    score += 0.05
                else:
                    score -= 0.05
            
            scores.append(max(0, min(1, score)))
        
        return scores
    
    def _calculate_diversity_score(self, smiles_list):
        """计算分子多样性得分"""
        try:
            from rdkit.Chem import rdMolDescriptors
            from rdkit import DataStructs
            
            fps = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
                else:
                    fps.append(None)
            
            diversity_scores = []
            for i, fp1 in enumerate(fps):
                if fp1 is None:
                    diversity_scores.append(0.0)
                    continue
                
                similarities = []
                for j, fp2 in enumerate(fps):
                    if i != j and fp2 is not None:
                        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                        similarities.append(sim)
                
                # 多样性 = 1 - 平均相似性
                if similarities:
                    diversity = 1 - np.mean(similarities)
                else:
                    diversity = 1.0
                
                diversity_scores.append(diversity)
            
            return diversity_scores
            
        except ImportError:
            # 如果没有相关库，返回随机多样性得分
            return np.random.uniform(0.3, 0.8, len(smiles_list))
    
    def get_top_hits(self, smiles_list, top_n=20, min_activity_score=0.7, max_lipinski_violations=1):
        """获取top-hit化合物"""
        print(f"\n=== Top-Hit化合物推荐系统 ===")
        print(f"输入化合物数量: {len(smiles_list)}")
        
        # 排序化合物
        ranked_results = self.rank_compounds(smiles_list)
        
        # 应用过滤条件
        filtered_results = ranked_results[
            (ranked_results['Activity_Score'] >= min_activity_score) &
            (ranked_results['Lipinski_Violations'] <= max_lipinski_violations)
        ]
        
        print(f"通过过滤条件的化合物数量: {len(filtered_results)}")
        
        # 获取top-N
        top_hits = filtered_results.head(top_n)
        
        print(f"\n=== Top {len(top_hits)} Hit化合物 ===")
        for i, (_, row) in enumerate(top_hits.iterrows(), 1):
            print(f"{i:2d}. 活性得分: {row['Activity_Score']:.3f}, "
                  f"置信度: {row['Confidence']:.3f}, "
                  f"综合得分: {row['Composite_Score']:.3f}")
            print(f"    SMILES: {row['SMILES']}")
            print(f"    分子量: {row['MW']:.1f}, LogP: {row['LogP']:.2f}, "
                  f"Lipinski违反: {row['Lipinski_Violations']:.0f}")
            print()
        
        return top_hits
    
    def visualize_results(self, results, save_path='top_hit_analysis.png'):
        """可视化推荐结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Top-Hit化合物推荐分析\nTop-Hit Compound Recommendation Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. 活性得分分布
        axes[0, 0].hist(results['Activity_Score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(0.7, color='red', linestyle='--', label='推荐阈值 (0.7)')
        axes[0, 0].set_title('活性得分分布\nActivity Score Distribution')
        axes[0, 0].set_xlabel('活性得分')
        axes[0, 0].set_ylabel('化合物数量')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 分子量 vs LogP
        scatter = axes[0, 1].scatter(results['MW'], results['LogP'], 
                                   c=results['Activity_Score'], cmap='viridis', alpha=0.7)
        axes[0, 1].axhline(5, color='red', linestyle='--', alpha=0.7, label='LogP=5')
        axes[0, 1].axvline(500, color='red', linestyle='--', alpha=0.7, label='MW=500')
        axes[0, 1].set_title('分子量 vs LogP\nMolecular Weight vs LogP')
        axes[0, 1].set_xlabel('分子量 (Da)')
        axes[0, 1].set_ylabel('LogP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='活性得分')
        
        # 3. Lipinski规则违反分布
        violation_counts = results['Lipinski_Violations'].value_counts().sort_index()
        axes[0, 2].bar(violation_counts.index, violation_counts.values, 
                      color=['green', 'yellow', 'orange', 'red', 'darkred'][:len(violation_counts)])
        axes[0, 2].set_title('Lipinski规则违反分布\nLipinski Rule Violations')
        axes[0, 2].set_xlabel('违反数量')
        axes[0, 2].set_ylabel('化合物数量')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Top-20化合物的综合得分
        top_20 = results.head(20)
        axes[1, 0].barh(range(len(top_20)), top_20['Composite_Score'], color='lightcoral')
        axes[1, 0].set_title('Top-20化合物综合得分\nTop-20 Composite Scores')
        axes[1, 0].set_xlabel('综合得分')
        axes[1, 0].set_ylabel('排名')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 活性得分 vs 置信度
        axes[1, 1].scatter(results['Activity_Score'], results['Confidence'], 
                          alpha=0.6, color='purple')
        axes[1, 1].set_title('活性得分 vs 置信度\nActivity Score vs Confidence')
        axes[1, 1].set_xlabel('活性得分')
        axes[1, 1].set_ylabel('置信度')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 药物相似性得分分布
        axes[1, 2].hist(results['Drug_Likeness_Score'], bins=20, alpha=0.7, 
                       color='lightgreen', edgecolor='black')
        axes[1, 2].set_title('药物相似性得分分布\nDrug-Likeness Score Distribution')
        axes[1, 2].set_xlabel('药物相似性得分')
        axes[1, 2].set_ylabel('化合物数量')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n可视化结果已保存为: {save_path}")
    
    def export_results(self, results, filename='top_hit_compounds.csv'):
        """导出结果到CSV文件"""
        # 选择重要列
        export_columns = [
            'Rank', 'SMILES', 'Activity_Score', 'Confidence', 'Composite_Score',
            'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'Lipinski_Violations',
            'Drug_Likeness_Score', 'Diversity_Score'
        ]
        
        export_data = results[export_columns].round(4)
        export_data.to_csv(filename, index=False)
        print(f"结果已导出到: {filename}")
        
        return export_data

# 示例使用函数
def demo_top_hit_recommendation():
    """演示Top-Hit推荐系统"""
    print("=== Top-Hit化合物推荐系统演示 ===")
    
    # 示例SMILES列表（实际使用时替换为真实的化合物库）
    example_smiles = [
        'CCO',  # 乙醇
        'CC(=O)O',  # 乙酸
        'c1ccccc1',  # 苯
        'CCN(CC)CC',  # 三乙胺
        'CC(C)O',  # 异丙醇
        'CCCCO',  # 丁醇
        'c1ccc(cc1)O',  # 苯酚
        'CC(=O)Nc1ccc(cc1)O',  # 对乙酰氨基酚
        'CC(C)(C)c1ccc(cc1)O',  # 4-叔丁基苯酚
        'Cc1ccc(cc1)N',  # 对甲苯胺
        'CCc1ccccc1',  # 乙苯
        'CC(=O)c1ccccc1',  # 苯乙酮
        'c1ccc2ccccc2c1',  # 萘
        'CCOc1ccccc1',  # 苯乙醚
        'Cc1cccc(c1)O',  # 间甲酚
        'CC(C)c1ccccc1',  # 异丙苯
        'c1ccc(cc1)C(=O)O',  # 苯甲酸
        'CCc1ccc(cc1)O',  # 4-乙基苯酚
        'Cc1ccc(cc1)C',  # 对二甲苯
        'c1ccc(cc1)N'  # 苯胺
    ]
    
    print(f"\n使用 {len(example_smiles)} 个示例化合物进行演示")
    print("注意：实际使用时需要加载训练好的模型和蛋白质数据")
    
    # 这里只是演示数据结构，实际使用时需要真实的模型
    print("\n演示数据结构:")
    print("1. 加载训练好的模型")
    print("2. 加载蛋白质结构数据")
    print("3. 创建推荐系统实例")
    print("4. 批量预测化合物活性")
    print("5. 计算药物性质")
    print("6. 综合排序和筛选")
    print("7. 生成Top-Hit推荐列表")
    print("8. 可视化分析结果")
    
    return example_smiles

if __name__ == "__main__":
    demo_top_hit_recommendation()