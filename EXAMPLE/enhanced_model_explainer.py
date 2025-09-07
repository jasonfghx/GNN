import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: SHAP库未安装，SHAP分析功能将不可用")

class EnhancedModelExplainer:
    """增强的模型解释系统"""
    
    def __init__(self, model, protein_data, device='cpu'):
        self.model = model
        self.protein_data = protein_data
        self.device = device
        self.model.eval()
        
        # 初始化GNNExplainer
        self.gnn_explainer = None
        self._setup_gnn_explainer()
        
        # 存储特征嵌入
        self.feature_embeddings = {}
        self.attention_weights = {}
        
    def _setup_gnn_explainer(self):
        """设置GNNExplainer"""
        try:
            # 创建GNNExplainer实例
            self.gnn_explainer = GNNExplainer(
                epochs=200,
                return_type='log_prob',
                feat_mask_type='scalar',
                edge_mask_type='object'
            )
            print("GNNExplainer初始化成功")
        except Exception as e:
            print(f"GNNExplainer初始化失败: {e}")
            self.gnn_explainer = None
    
    def extract_molecular_features(self, mol_data_list, layer_names=None):
        """提取分子特征嵌入"""
        if layer_names is None:
            layer_names = ['molecular_gnn', 'fusion_layer', 'predictor']
        
        features = {name: [] for name in layer_names}
        
        # 注册钩子函数
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    features[name].append(output.detach().cpu().numpy())
                elif isinstance(output, tuple):
                    features[name].append(output[0].detach().cpu().numpy())
            return hook
        
        # 为指定层注册钩子
        for name in layer_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                hook = layer.register_forward_hook(get_activation(name))
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            for mol_data in mol_data_list:
                if mol_data is not None:
                    batch = Batch.from_data_list([mol_data]).to(self.device)
                    _ = self.model(batch, self.protein_data)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 转换为numpy数组
        for name in features:
            if features[name]:
                features[name] = np.vstack(features[name])
            else:
                features[name] = np.array([])
        
        return features
    
    def explain_with_gnn_explainer(self, mol_data, target_class=1):
        """使用GNNExplainer解释预测"""
        if self.gnn_explainer is None:
            print("GNNExplainer不可用")
            return None, None
        
        try:
            # 准备数据
            batch = Batch.from_data_list([mol_data]).to(self.device)
            
            # 使用GNNExplainer
            node_mask, edge_mask = self.gnn_explainer.explain_graph(
                self.model, batch.x, batch.edge_index, 
                target=target_class
            )
            
            return node_mask.cpu().numpy(), edge_mask.cpu().numpy()
            
        except Exception as e:
            print(f"GNNExplainer解释失败: {e}")
            return None, None
    
    def explain_with_shap(self, smiles_list, background_size=50):
        """使用SHAP解释模型预测"""
        if not SHAP_AVAILABLE:
            print("SHAP库不可用")
            return None
        
        try:
            from dengue_ns5_inhibitor_prediction import MolecularGraphDataset
            
            # 创建数据集
            dummy_labels = [0] * len(smiles_list)
            dataset = MolecularGraphDataset(smiles_list, dummy_labels)
            
            # 准备背景数据
            background_indices = np.random.choice(
                len(dataset), 
                min(background_size, len(dataset)), 
                replace=False
            )
            
            def model_predict(indices):
                predictions = []
                for idx in indices:
                    mol_data, _ = dataset[int(idx)]
                    if mol_data is not None:
                        batch = Batch.from_data_list([mol_data]).to(self.device)
                        with torch.no_grad():
                            output = self.model(batch, self.protein_data)
                            predictions.append(output.cpu().numpy()[0])
                    else:
                        predictions.append(0.0)
                return np.array(predictions)
            
            # 创建SHAP解释器
            explainer = shap.KernelExplainer(
                model_predict, 
                background_indices
            )
            
            # 计算SHAP值
            test_indices = np.arange(len(dataset))
            shap_values = explainer.shap_values(test_indices)
            
            return shap_values
            
        except Exception as e:
            print(f"SHAP解释失败: {e}")
            return None
    
    def analyze_attention_weights(self, mol_data_list):
        """分析注意力权重"""
        attention_weights = []
        
        # 注册钩子获取注意力权重
        def get_attention_weights(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_weights.append(module.attention_weights.detach().cpu().numpy())
        
        hooks = []
        # 为注意力层注册钩子
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                hook = module.register_forward_hook(get_attention_weights)
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            for mol_data in mol_data_list:
                if mol_data is not None:
                    batch = Batch.from_data_list([mol_data]).to(self.device)
                    _ = self.model(batch, self.protein_data)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def perform_tsne_analysis(self, features, labels=None, perplexity=30, n_components=2):
        """执行t-SNE降维分析"""
        print(f"执行t-SNE分析，特征维度: {features.shape}")
        
        # 标准化特征
        features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # t-SNE降维
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, features.shape[0] - 1),
            random_state=42,
            n_iter=1000
        )
        
        tsne_features = tsne.fit_transform(features_normalized)
        
        return tsne_features
    
    def perform_umap_analysis(self, features, labels=None, n_neighbors=15, n_components=2):
        """执行UMAP降维分析"""
        print(f"执行UMAP分析，特征维度: {features.shape}")
        
        try:
            # 标准化特征
            features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            # UMAP降维
            umap_reducer = umap.UMAP(
                n_neighbors=min(n_neighbors, features.shape[0] - 1),
                n_components=n_components,
                random_state=42
            )
            
            umap_features = umap_reducer.fit_transform(features_normalized)
            
            return umap_features
            
        except Exception as e:
            print(f"UMAP分析失败: {e}")
            return None
    
    def visualize_molecular_explanation(self, smiles, node_importance=None, save_path=None):
        """可视化分子解释"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无法解析SMILES: {smiles}")
                return None
            
            # 生成2D坐标
            rdDepictor.Compute2DCoords(mol)
            
            # 创建绘图器
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            
            # 设置原子颜色（基于重要性）
            if node_importance is not None:
                # 归一化重要性分数
                importance_norm = (node_importance - node_importance.min()) / \
                                (node_importance.max() - node_importance.min() + 1e-8)
                
                # 创建颜色映射
                atom_colors = {}
                for i, importance in enumerate(importance_norm):
                    # 红色表示高重要性，蓝色表示低重要性
                    red = importance
                    blue = 1 - importance
                    atom_colors[i] = (red, 0.0, blue)
                
                drawer.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())),
                                  highlightAtomColors=atom_colors)
            else:
                drawer.DrawMolecule(mol)
            
            drawer.FinishDrawing()
            
            # 保存图像
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(drawer.GetDrawingText())
                print(f"分子解释图已保存: {save_path}")
            
            return drawer.GetDrawingText()
            
        except Exception as e:
            print(f"分子可视化失败: {e}")
            return None
    
    def comprehensive_analysis(self, smiles_list, labels=None, save_dir='explanation_results'):
        """综合分析"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("=== 开始综合模型解释分析 ===")
        
        # 1. 准备数据
        from dengue_ns5_inhibitor_prediction import MolecularGraphDataset
        dummy_labels = labels if labels is not None else [0] * len(smiles_list)
        dataset = MolecularGraphDataset(smiles_list, dummy_labels)
        
        mol_data_list = []
        valid_indices = []
        for i, (mol_data, _) in enumerate(dataset):
            if mol_data is not None:
                mol_data_list.append(mol_data)
                valid_indices.append(i)
        
        print(f"有效分子数量: {len(mol_data_list)} / {len(smiles_list)}")
        
        # 2. 提取特征嵌入
        print("\n提取分子特征嵌入...")
        features = self.extract_molecular_features(mol_data_list)
        
        # 3. 降维分析
        results = {}
        
        for layer_name, layer_features in features.items():
            if layer_features.size > 0:
                print(f"\n分析 {layer_name} 层特征...")
                
                # t-SNE分析
                tsne_features = self.perform_tsne_analysis(layer_features)
                results[f'{layer_name}_tsne'] = tsne_features
                
                # UMAP分析
                umap_features = self.perform_umap_analysis(layer_features)
                if umap_features is not None:
                    results[f'{layer_name}_umap'] = umap_features
                
                # PCA分析
                pca = PCA(n_components=2)
                pca_features = pca.fit_transform(layer_features)
                results[f'{layer_name}_pca'] = pca_features
                
                print(f"  t-SNE方差解释: {tsne_features.var(axis=0).sum():.3f}")
                if umap_features is not None:
                    print(f"  UMAP方差解释: {umap_features.var(axis=0).sum():.3f}")
                print(f"  PCA方差解释: {pca.explained_variance_ratio_.sum():.3f}")
        
        # 4. 可视化结果
        self.visualize_comprehensive_results(results, valid_indices, labels, save_dir)
        
        # 5. GNNExplainer分析（示例）
        if len(mol_data_list) > 0:
            print("\n执行GNNExplainer分析...")
            node_mask, edge_mask = self.explain_with_gnn_explainer(mol_data_list[0])
            if node_mask is not None:
                # 可视化第一个分子的解释
                self.visualize_molecular_explanation(
                    smiles_list[valid_indices[0]], 
                    node_mask,
                    os.path.join(save_dir, 'molecule_explanation_example.png')
                )
        
        # 6. SHAP分析
        if SHAP_AVAILABLE and len(smiles_list) <= 100:  # 限制SHAP分析的样本数量
            print("\n执行SHAP分析...")
            shap_values = self.explain_with_shap(smiles_list[:50])  # 限制为前50个
            if shap_values is not None:
                results['shap_values'] = shap_values
        
        print(f"\n=== 分析完成，结果保存在: {save_dir} ===")
        return results
    
    def visualize_comprehensive_results(self, results, indices, labels, save_dir):
        """可视化综合分析结果"""
        # 创建大型图表
        n_methods = len([k for k in results.keys() if not k.startswith('shap')])
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(labels)) if labels else 2))
        
        for key, features in results.items():
            if key.startswith('shap'):
                continue
                
            if plot_idx >= n_rows * n_cols:
                break
                
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            # 绘制散点图
            if labels is not None:
                valid_labels = [labels[i] for i in indices]
                unique_labels = list(set(valid_labels))
                for i, label in enumerate(unique_labels):
                    mask = np.array(valid_labels) == label
                    ax.scatter(features[mask, 0], features[mask, 1], 
                             c=[colors[i]], label=f'类别 {label}', alpha=0.7)
                ax.legend()
            else:
                ax.scatter(features[:, 0], features[:, 1], alpha=0.7)
            
            ax.set_title(f'{key.replace("_", " ").title()}')
            ax.set_xlabel('维度 1')
            ax.set_ylabel('维度 2')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # 隐藏多余的子图
        for i in range(plot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存特征重要性分析
        if 'shap_values' in results:
            self.plot_shap_summary(results['shap_values'], save_dir)
    
    def plot_shap_summary(self, shap_values, save_dir):
        """绘制SHAP摘要图"""
        if not SHAP_AVAILABLE:
            return
            
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, show=False)
            plt.title('SHAP特征重要性分析\nSHAP Feature Importance Analysis')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"SHAP可视化失败: {e}")

# 示例使用函数
def demo_model_explanation():
    """演示模型解释功能"""
    print("=== 增强模型解释系统演示 ===")
    
    # 示例SMILES
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
        'Cc1ccc(cc1)N'  # 对甲苯胺
    ]
    
    example_labels = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]  # 示例标签
    
    print(f"\n使用 {len(example_smiles)} 个示例化合物进行演示")
    print("\n功能演示:")
    print("1. 特征嵌入提取")
    print("2. t-SNE/UMAP/PCA降维分析")
    print("3. GNNExplainer原子重要性分析")
    print("4. SHAP特征重要性分析")
    print("5. 注意力权重可视化")
    print("6. 分子结构解释可视化")
    print("7. 综合分析报告生成")
    
    return example_smiles, example_labels

if __name__ == "__main__":
    demo_model_explanation()