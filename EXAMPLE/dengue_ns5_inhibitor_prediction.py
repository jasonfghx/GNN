import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 尝试导入SHAP，如果没有安装则跳过
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP未安装，将跳过SHAP分析")

# 导入其他必要库
from torch_geometric.utils import to_networkx
import networkx as nx
from collections import defaultdict

class MolecularGraphDataset(Dataset):
    """分子图数据集类"""
    def __init__(self, smiles_list, labels, protein_features=None):
        self.smiles_list = smiles_list
        self.labels = labels
        self.protein_features = protein_features
        self.mol_graphs = []
        
        # 预处理分子图
        for smiles in smiles_list:
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                self.mol_graphs.append(graph)
            else:
                # 如果无法解析SMILES，创建空图
                self.mol_graphs.append(self.create_empty_graph())
    
    def smiles_to_graph(self, smiles):
        """将SMILES转换为分子图"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 获取原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    atom.GetTotalNumHs()
                ]
                atom_features.append(features)
            
            # 获取边信息
            edge_indices = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])  # 无向图
                
                bond_features = [
                    int(bond.GetBondType()),
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing())
                ]
                edge_features.extend([bond_features, bond_features])
            
            # 转换为PyTorch张量
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        except:
            return None
    
    def create_empty_graph(self):
        """创建空图用于无效SMILES"""
        x = torch.zeros((1, 6), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        return self.mol_graphs[idx], torch.tensor(self.labels[idx], dtype=torch.float)

class ProteinGraphProcessor:
    """蛋白质图处理器"""
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.protein_graph = self.process_protein_structure()
    
    def process_protein_structure(self):
        """处理PDB结构文件"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', self.pdb_file)
            
            # 提取残基信息
            residues = []
            coordinates = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == ' ':  # 只考虑标准残基
                            try:
                                ca_atom = residue['CA']  # α碳原子
                                coordinates.append(ca_atom.get_coord())
                                residues.append(residue.get_resname())
                            except KeyError:
                                continue
            
            # 创建残基特征
            residue_features = self.encode_residues(residues)
            
            # 基于距离创建边
            edge_indices, edge_features = self.create_edges_from_distance(
                coordinates, threshold=8.0
            )
            
            x = torch.tensor(residue_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        except:
            # 如果处理失败，返回简化的蛋白质特征
            return self.create_simplified_protein_graph()
    
    def encode_residues(self, residues):
        """编码残基特征"""
        # 20种标准氨基酸的简化编码
        aa_dict = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }
        
        features = []
        for residue in residues:
            # 创建one-hot编码
            feature = [0] * 20
            if residue in aa_dict:
                feature[aa_dict[residue]] = 1
            features.append(feature)
        
        return features
    
    def create_edges_from_distance(self, coordinates, threshold=8.0):
        """基于距离创建边"""
        coordinates = np.array(coordinates)
        n_residues = len(coordinates)
        
        edge_indices = []
        edge_features = []
        
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance <= threshold:
                    edge_indices.extend([[i, j], [j, i]])
                    edge_features.extend([[distance], [distance]])
        
        return edge_indices, edge_features
    
    def create_simplified_protein_graph(self):
        """创建简化的蛋白质图"""
        # 创建一个简化的蛋白质表示
        x = torch.randn(100, 20)  # 100个残基，每个20维特征
        edge_index = torch.randint(0, 100, (2, 200))  # 随机边
        edge_attr = torch.randn(200, 1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""
    def __init__(self, mol_dim, protein_dim, hidden_dim=128):
        super().__init__()
        self.mol_dim = mol_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        
        self.mol_proj = nn.Linear(mol_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
    def forward(self, mol_features, protein_features):
        # 投影到相同维度
        mol_proj = self.mol_proj(mol_features.unsqueeze(1))  # [batch, 1, hidden_dim]
        protein_proj = self.protein_proj(protein_features.unsqueeze(1))  # [batch, 1, hidden_dim]
        
        # 交叉注意力
        attended_mol, _ = self.attention(mol_proj, protein_proj, protein_proj)
        attended_protein, _ = self.attention(protein_proj, mol_proj, mol_proj)
        
        # 融合特征
        fused_features = torch.cat([
            attended_mol.squeeze(1),
            attended_protein.squeeze(1)
        ], dim=1)
        
        return fused_features

class JointGraphFusion(nn.Module):
    """联合图融合模块"""
    def __init__(self, mol_dim=6, protein_dim=20, hidden_dim=128):
        super().__init__()
        self.mol_dim = mol_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        
        # 统一特征维度
        self.mol_proj = nn.Linear(mol_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        # 联合图卷积层
        self.joint_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.joint_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.joint_conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.2)
        
    def create_joint_graph(self, mol_data, protein_data, batch_size):
        """创建分子-蛋白质联合图"""
        joint_graphs = []
        
        for i in range(batch_size):
            # 获取分子图节点和边
            mol_start = (mol_data.batch == i).nonzero(as_tuple=True)[0]
            if len(mol_start) == 0:
                continue
                
            mol_nodes = mol_data.x[mol_start]
            mol_edges = mol_data.edge_index[:, (mol_data.batch[mol_data.edge_index[0]] == i)]
            
            # 投影分子和蛋白质特征到相同维度
            mol_features = self.mol_proj(mol_nodes)
            protein_features = self.protein_proj(protein_data.x)
            
            # 合并节点特征
            num_mol_nodes = mol_features.size(0)
            num_protein_nodes = protein_features.size(0)
            
            joint_features = torch.cat([mol_features, protein_features], dim=0)
            
            # 调整分子边索引
            mol_edges_adjusted = mol_edges - mol_start[0]
            
            # 调整蛋白质边索引
            protein_edges_adjusted = protein_data.edge_index + num_mol_nodes
            
            # 创建分子-蛋白质连接边（简化版本：连接所有分子原子到蛋白质质心）
            protein_center_idx = num_mol_nodes + num_protein_nodes // 2  # 简化的质心
            mol_protein_edges = torch.stack([
                torch.arange(num_mol_nodes),
                torch.full((num_mol_nodes,), protein_center_idx)
            ])
            
            # 合并所有边
            joint_edges = torch.cat([
                mol_edges_adjusted,
                protein_edges_adjusted,
                mol_protein_edges,
                mol_protein_edges.flip(0)  # 双向边
            ], dim=1)
            
            joint_graphs.append(Data(x=joint_features, edge_index=joint_edges))
        
        return Batch.from_data_list(joint_graphs)
    
    def forward(self, mol_data, protein_data, batch_size):
        # 创建联合图
        joint_graph = self.create_joint_graph(mol_data, protein_data, batch_size)
        
        # 联合图卷积
        x = joint_graph.x
        edge_index = joint_graph.edge_index
        
        x = F.relu(self.joint_conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.joint_conv2(x, edge_index))
        x = self.dropout(x)
        x = self.joint_conv3(x, edge_index)
        
        # 全局池化
        batch_tensor = joint_graph.batch if hasattr(joint_graph, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled_features = global_mean_pool(x, batch_tensor)
        
        return pooled_features

class MolecularGNN(nn.Module):
    """分子图神经网络"""
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # 全局池化
        x = global_mean_pool(x, batch)
        return x

class ProteinGNN(nn.Module):
    """蛋白质图神经网络"""
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=128):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1, concat=False)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # 全局平均池化
        x = torch.mean(x, dim=0, keepdim=True)
        return x

class DengueNS5InhibitorPredictor(nn.Module):
    """登革热NS5抑制剂预测模型"""
    def __init__(self, fusion_strategy='cross_attention'):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        
        # 分子和蛋白质编码器
        if fusion_strategy != 'joint_graph':
            self.mol_encoder = MolecularGNN()
            self.protein_encoder = ProteinGNN()
        
        # 融合策略
        if fusion_strategy == 'concat':
            self.fusion_dim = 256  # 128 + 128
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        elif fusion_strategy == 'cross_attention':
            self.fusion = CrossAttentionFusion(128, 128)
            self.fusion_dim = 256
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        elif fusion_strategy == 'joint_graph':
            self.joint_fusion = JointGraphFusion()
            self.fusion_dim = 128
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def forward(self, mol_data, protein_data):
        if self.fusion_strategy == 'joint_graph':
            # 联合图融合
            batch_size = mol_data.batch.max().item() + 1 if hasattr(mol_data, 'batch') else 1
            fused_features = self.joint_fusion(mol_data, protein_data, batch_size)
        else:
            # 编码分子和蛋白质特征
            mol_features = self.mol_encoder(mol_data)
            protein_features = self.protein_encoder(protein_data)
            
            # 特征融合
            if self.fusion_strategy == 'concat':
                # 简单拼接
                protein_features_expanded = protein_features.expand(mol_features.size(0), -1)
                fused_features = torch.cat([mol_features, protein_features_expanded], dim=1)
            elif self.fusion_strategy == 'cross_attention':
                # 交叉注意力融合
                protein_features_expanded = protein_features.expand(mol_features.size(0), -1)
                fused_features = self.fusion(mol_features, protein_features_expanded)
        
        # 分类预测
        output = self.classifier(fused_features)
        return output.squeeze()

def load_and_preprocess_data():
    """加载和预处理数据"""
    try:
        # 加载活性化合物数据
        active_df = pd.read_csv('DENV inhibitors RdRp_登革熱病毒抑制物_3739(2).csv', 
                               sep=';', encoding='utf-8', on_bad_lines='skip')
        
        # 加载非活性化合物数据
        inactive_df = pd.read_csv('inactive compounds_無活性抑制物(1).csv', 
                                 sep=';', encoding='utf-8', on_bad_lines='skip')
        
        # 提取SMILES和标签
        active_smiles = active_df['Smiles'].dropna().tolist()
        inactive_smiles = inactive_df['Smiles'].dropna().tolist()
        
        # 过滤有效的SMILES
        valid_active_smiles = []
        for smiles in active_smiles:
            if isinstance(smiles, str) and len(smiles) > 5:
                valid_active_smiles.append(smiles)
        
        valid_inactive_smiles = []
        for smiles in inactive_smiles:
            if isinstance(smiles, str) and len(smiles) > 5:
                valid_inactive_smiles.append(smiles)
        
        # 创建标签（1为活性，0为非活性）
        active_labels = [1] * len(valid_active_smiles)
        inactive_labels = [0] * len(valid_inactive_smiles)
        
        # 合并数据
        all_smiles = valid_active_smiles + valid_inactive_smiles
        all_labels = active_labels + inactive_labels
        
        print(f"活性化合物数量: {len(valid_active_smiles)}")
        print(f"非活性化合物数量: {len(valid_inactive_smiles)}")
        print(f"总化合物数量: {len(all_smiles)}")
        
        return all_smiles, all_labels
        
    except Exception as e:
        print(f"数据加载错误: {e}")
        # 如果数据加载失败，创建一些示例数据用于测试
        print("使用示例数据进行测试...")
        example_smiles = [
            'CCO',  # 乙醇
            'CC(=O)O',  # 乙酸
            'c1ccccc1',  # 苯
            'CCN(CC)CC',  # 三乙胺
            'CC(C)O',  # 异丙醇
            'CCCCO',  # 丁醇
            'c1ccc(cc1)O',  # 苯酚
            'CC(=O)Nc1ccc(cc1)O'  # 对乙酰氨基酚
        ]
        example_labels = [1, 1, 0, 0, 1, 0, 1, 0]
        
        print(f"示例数据 - 活性化合物: 4, 非活性化合物: 4, 总数: 8")
        return example_smiles, example_labels

def train_model(model, train_loader, val_loader, protein_data, epochs=100, model_save_path='best_dengue_ns5_model.pth'):
    """训练模型"""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_mol, batch_labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_mol, protein_data)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch_mol, batch_labels in val_loader:
                outputs = model(batch_mol, protein_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_true_labels.extend(batch_labels.cpu().numpy())
        
        # 计算指标
        val_pred_binary = (np.array(val_predictions) > 0.5).astype(int)
        val_accuracy = accuracy_score(val_true_labels, val_pred_binary)
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return train_losses, val_losses, val_accuracies

def cross_validate_model(dataset, protein_data, fusion_strategy='cross_attention', k_folds=5, epochs=50):
    """K-fold交叉验证"""
    print(f"\n=== {k_folds}-Fold 交叉验证 ({fusion_strategy}) ===")
    
    # 获取标签用于分层抽样
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    
    # 创建分层K-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    def collate_fn(batch):
        mol_graphs, labels = zip(*batch)
        return Batch.from_data_list(mol_graphs), torch.tensor(labels, dtype=torch.float)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        
        # 创建数据加载器
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        # 创建新模型
        model = DengueNS5InhibitorPredictor(fusion_strategy=fusion_strategy)
        
        # 训练模型
        model_save_path = f'fold_{fold+1}_{fusion_strategy}_model.pth'
        train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, protein_data, epochs=epochs, model_save_path=model_save_path
        )
        
        # 加载最佳模型并评估
        model.load_state_dict(torch.load(model_save_path))
        fold_result = evaluate_model(model, val_loader, protein_data)
        fold_results.append(fold_result)
        
        print(f"Fold {fold + 1} 结果: Accuracy={fold_result['accuracy']:.4f}, "
              f"ROC-AUC={fold_result['roc_auc']:.4f}, F1={fold_result['f1']:.4f}")
    
    # 计算平均结果
    avg_results = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'roc_auc': np.mean([r['roc_auc'] for r in fold_results]),
        'pr_auc': np.mean([r['pr_auc'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
        'std_roc_auc': np.std([r['roc_auc'] for r in fold_results]),
        'std_pr_auc': np.std([r['pr_auc'] for r in fold_results]),
        'std_f1': np.std([r['f1'] for r in fold_results])
    }
    
    print(f"\n=== {k_folds}-Fold 交叉验证平均结果 ===")
    print(f"Accuracy: {avg_results['accuracy']:.4f} ± {avg_results['std_accuracy']:.4f}")
    print(f"ROC-AUC: {avg_results['roc_auc']:.4f} ± {avg_results['std_roc_auc']:.4f}")
    print(f"PR-AUC: {avg_results['pr_auc']:.4f} ± {avg_results['std_pr_auc']:.4f}")
    print(f"F1-Score: {avg_results['f1']:.4f} ± {avg_results['std_f1']:.4f}")
    
    return avg_results, fold_results

def evaluate_model(model, test_loader, protein_data):
    """评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch_mol, batch_labels in test_loader:
            outputs = model(batch_mol, protein_data)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    pred_binary = (predictions > 0.5).astype(int)
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, pred_binary)
    roc_auc = roc_auc_score(true_labels, predictions)
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    pr_auc = auc(recall, precision)
    f1 = f1_score(true_labels, pred_binary)
    
    print(f"\n=== 模型评估结果 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def plot_results(train_losses, val_losses, val_accuracies, eval_results):
    """绘制结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 训练曲线
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 验证准确率
    axes[0, 1].plot(val_accuracies)
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)
    
    # 混淆矩阵
    pred_binary = (eval_results['predictions'] > 0.5).astype(int)
    cm = confusion_matrix(eval_results['true_labels'], pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_ylabel('True Label')
    
    # ROC曲线
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(eval_results['true_labels'], eval_results['predictions'])
    axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {eval_results["roc_auc"]:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('dengue_ns5_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(model, test_loader, protein_data, save_plots=True):
    """增强的特征重要性分析"""
    print("\n=== 特征重要性分析 ===")
    
    model.eval()
    
    # 1. 原子类型重要性分析
    atom_importance = analyze_atom_importance(model, test_loader, protein_data)
    
    # 2. 梯度分析
    gradient_importance = analyze_gradient_importance(model, test_loader, protein_data)
    
    # 3. SHAP分析（如果可用）
    if SHAP_AVAILABLE:
        try:
            shap_analysis = analyze_shap_importance(model, test_loader, protein_data)
        except Exception as e:
            print(f"SHAP分析失败: {e}")
            shap_analysis = None
    else:
        shap_analysis = None
    
    # 4. 注意力权重分析（对于cross_attention策略）
    if hasattr(model, 'fusion') and hasattr(model.fusion, 'attention'):
        attention_weights = analyze_attention_weights(model, test_loader, protein_data)
    else:
        attention_weights = None
    
    # 5. 可视化结果
    if save_plots:
        plot_feature_importance_results(atom_importance, gradient_importance, 
                                       shap_analysis, attention_weights)
    
    return {
        'atom_importance': atom_importance,
        'gradient_importance': gradient_importance,
        'shap_analysis': shap_analysis,
        'attention_weights': attention_weights
    }

def analyze_atom_importance(model, test_loader, protein_data):
    """分析原子类型重要性"""
    print("\n--- 原子类型重要性分析 ---")
    
    atom_contributions = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for batch_mol, batch_labels in test_loader:
            # 获取原子特征
            atom_features = batch_mol.x
            atomic_nums = atom_features[:, 0].int()  # 原子序数
            
            # 计算预测
            outputs = model(batch_mol, protein_data)
            
            # 按原子类型统计贡献
            for i, atomic_num in enumerate(atomic_nums):
                atom_type = atomic_num.item()
                batch_idx = batch_mol.batch[i].item() if hasattr(batch_mol, 'batch') else 0
                if batch_idx < len(outputs):
                    atom_contributions[atom_type].append(outputs[batch_idx].item())
    
    # 计算平均贡献
    avg_contributions = {}
    for atom_type, contributions in atom_contributions.items():
        avg_contributions[atom_type] = np.mean(contributions)
    
    # 原子类型映射
    atom_names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
    
    print("原子类型平均贡献:")
    for atom_type, contribution in sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True):
        atom_name = atom_names.get(atom_type, f'Atom_{atom_type}')
        print(f"{atom_name} ({atom_type}): {contribution:.4f}")
    
    return avg_contributions

def analyze_gradient_importance(model, test_loader, protein_data):
    """基于梯度的重要性分析"""
    print("\n--- 梯度重要性分析 ---")
    
    model.eval()
    gradient_importance = []
    
    for batch_mol, batch_labels in test_loader:
        batch_mol.x.requires_grad_(True)
        
        outputs = model(batch_mol, protein_data)
        loss = F.binary_cross_entropy(outputs, batch_labels)
        
        # 计算梯度
        loss.backward()
        
        # 收集梯度信息
        if batch_mol.x.grad is not None:
            grad_norm = torch.norm(batch_mol.x.grad, dim=1)
            gradient_importance.extend(grad_norm.detach().cpu().numpy())
        
        # 清除梯度
        model.zero_grad()
        batch_mol.x.grad = None
    
    avg_gradient_importance = np.mean(gradient_importance) if gradient_importance else 0
    print(f"平均梯度重要性: {avg_gradient_importance:.4f}")
    
    return gradient_importance

def analyze_shap_importance(model, test_loader, protein_data):
    """SHAP重要性分析"""
    print("\n--- SHAP重要性分析 ---")
    
    # 创建SHAP解释器的包装函数
    def model_wrapper(x):
        # 这里需要根据具体的输入格式调整
        # 简化版本，实际使用时需要更复杂的处理
        with torch.no_grad():
            return model(x, protein_data).cpu().numpy()
    
    # 获取一小批数据用于SHAP分析
    sample_data = []
    for i, (batch_mol, batch_labels) in enumerate(test_loader):
        if i >= 3:  # 只取前3个batch
            break
        sample_data.append((batch_mol, batch_labels))
    
    print("SHAP分析需要更复杂的实现，当前提供框架")
    return None

def analyze_attention_weights(model, test_loader, protein_data):
    """分析注意力权重"""
    print("\n--- 注意力权重分析 ---")
    
    attention_weights = []
    
    model.eval()
    with torch.no_grad():
        for batch_mol, batch_labels in test_loader:
            # 获取分子和蛋白质特征
            if hasattr(model, 'mol_encoder'):
                mol_features = model.mol_encoder(batch_mol)
                protein_features = model.protein_encoder(protein_data)
                
                # 扩展蛋白质特征
                protein_features_expanded = protein_features.expand(mol_features.size(0), -1)
                
                # 获取注意力权重
                mol_proj = model.fusion.mol_proj(mol_features.unsqueeze(1))
                protein_proj = model.fusion.protein_proj(protein_features_expanded.unsqueeze(1))
                
                _, attn_weights = model.fusion.attention(mol_proj, protein_proj, protein_proj)
                attention_weights.append(attn_weights.cpu().numpy())
    
    if attention_weights:
        avg_attention = np.mean(np.concatenate(attention_weights, axis=0), axis=0)
        print(f"平均注意力权重形状: {avg_attention.shape}")
        print(f"注意力权重统计: min={avg_attention.min():.4f}, max={avg_attention.max():.4f}, mean={avg_attention.mean():.4f}")
    
    return attention_weights

def plot_feature_importance_results(atom_importance, gradient_importance, shap_analysis, attention_weights):
    """可视化特征重要性结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 原子类型重要性
    if atom_importance:
        atom_names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
        atoms = [atom_names.get(k, f'Atom_{k}') for k in atom_importance.keys()]
        importances = list(atom_importance.values())
        
        axes[0, 0].bar(atoms, importances)
        axes[0, 0].set_title('原子类型重要性')
        axes[0, 0].set_ylabel('平均贡献')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 梯度重要性分布
    if gradient_importance:
        axes[0, 1].hist(gradient_importance, bins=50, alpha=0.7)
        axes[0, 1].set_title('梯度重要性分布')
        axes[0, 1].set_xlabel('梯度范数')
        axes[0, 1].set_ylabel('频次')
    
    # 3. 注意力权重热图
    if attention_weights:
        avg_attention = np.mean(np.concatenate(attention_weights, axis=0), axis=0)
        im = axes[1, 0].imshow(avg_attention, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('平均注意力权重')
        plt.colorbar(im, ax=axes[1, 0])
    
    # 4. 模型层级重要性（占位）
    axes[1, 1].text(0.5, 0.5, 'SHAP分析\n(需要进一步实现)', 
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('SHAP重要性')
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n特征重要性分析图表已保存为 'feature_importance_analysis.png'")

def main():
    """主函数"""
    print("=== 登革热病毒NS5抑制剂预测模型 ===")
    
    # 1. 数据加载和预处理
    print("\n1. 加载和预处理数据...")
    smiles_list, labels = load_and_preprocess_data()
    
    # 2. 处理蛋白质结构
    print("\n2. 处理蛋白质结构...")
    protein_processor = ProteinGraphProcessor('Dengue virus 3序列.pdb')
    protein_data = protein_processor.protein_graph
    
    # 3. 创建数据集
    print("\n3. 创建分子图数据集...")
    dataset = MolecularGraphDataset(smiles_list, labels)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"正样本比例: {np.mean(labels):.3f}")
    
    # 创建数据加载器函数
    def collate_fn(batch):
        mol_graphs, labels = zip(*batch)
        return Batch.from_data_list(mol_graphs), torch.tensor(labels, dtype=torch.float)
    
    # 可以尝试不同的融合策略
    strategies = ['concat', 'cross_attention', 'joint_graph']
    
    # 选择运行模式
    run_cross_validation = False  # 设置为True进行交叉验证，False进行单次训练
    
    if run_cross_validation:
        print("\n=== 运行K-fold交叉验证 ===")
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"交叉验证 - 融合策略: {strategy}")
            print(f"{'='*60}")
            
            # 进行5-fold交叉验证
            cv_results, fold_results = cross_validate_model(
                dataset=dataset,
                protein_data=protein_data,
                fusion_strategy=strategy,
                k_folds=5,
                epochs=30  # 交叉验证时使用较少的epoch
            )
            
            print(f"\n{strategy} 策略交叉验证结果:")
            for metric, value in cv_results.items():
                if 'std_' not in metric:
                    std_metric = f'std_{metric}'
                    if std_metric in cv_results:
                        print(f"{metric}: {value:.4f} ± {cv_results[std_metric]:.4f}")
                    else:
                        print(f"{metric}: {value:.4f}")
    
    else:
        print("\n=== 运行单次训练验证 ===")
        
        # 4. 数据分割
        train_indices, test_indices = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42, stratify=labels
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.2, random_state=42, 
            stratify=[labels[i] for i in train_indices]
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 5. 创建和训练模型
        print("\n4. 创建和训练模型...")
        
        for strategy in strategies:
            print(f"\n=== 使用 {strategy} 融合策略 ===")
            
            model = DengueNS5InhibitorPredictor(fusion_strategy=strategy)
            
            # 训练模型
            model_save_path = f'best_dengue_ns5_model_{strategy}.pth'
            train_losses, val_losses, val_accuracies = train_model(
                model, train_loader, val_loader, protein_data, epochs=50, model_save_path=model_save_path
            )
            
            # 加载最佳模型
            model.load_state_dict(torch.load(model_save_path))
            
            # 评估模型
            eval_results = evaluate_model(model, test_loader, protein_data)
            
            # 绘制结果
            plot_results(train_losses, val_losses, val_accuracies, eval_results)
            
            # 特征重要性分析
            importance_results = analyze_feature_importance(model, test_loader, protein_data, save_plots=True)
            
            # 保存模型和结果
            torch.save({
                'model_state_dict': model.state_dict(),
                'fusion_strategy': strategy,
                'test_metrics': eval_results,
                'importance_results': importance_results,
                'data_split': {
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'test_indices': test_indices
                }
            }, f'dengue_ns5_model_{strategy}.pth')
            
            print(f"\n{strategy} 策略模型训练完成！")
    
    print("\n=== 模型训练和评估完成 ===")
    print("\n输出文件:")
    if run_cross_validation:
        for strategy in strategies:
            print(f"- fold_*_{strategy}_model.pth: {strategy}策略交叉验证模型")
    else:
        for strategy in strategies:
            print(f"- dengue_ns5_model_{strategy}.pth: {strategy}策略完整模型")
            print(f"- best_dengue_ns5_model_{strategy}.pth: {strategy}策略最佳权重")
    print("- dengue_ns5_model_results.png: 评估结果图表")
    print("- feature_importance_analysis.png: 特征重要性分析图表")
    
    # 生成实验报告
    generate_experiment_report(strategies, run_cross_validation)

def load_trained_model(model_path, device='cpu'):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint.get('model_config', {
        'mol_input_dim': 6,
        'protein_input_dim': 20,
        'hidden_dim': 128,
        'fusion_strategy': 'concat'
    })
    
    # 创建模型
    model = DengueNS5InhibitorPredictor(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def predict_inhibitor_activity(model, mol_data, protein_data, device='cpu'):
    """预测分子的抑制剂活性"""
    model.eval()
    
    with torch.no_grad():
        if isinstance(mol_data, list):
            # 批量预测
            predictions = []
            for mol in mol_data:
                mol = mol.to(device)
                pred = model(mol, protein_data.to(device))
                predictions.append(pred.cpu().numpy())
            return np.array(predictions)
        else:
            # 单个预测
            mol_data = mol_data.to(device)
            protein_data = protein_data.to(device)
            prediction = model(mol_data, protein_data)
            return prediction.cpu().numpy()

def compare_fusion_strategies(model_paths, test_data, protein_data, device='cpu'):
    """比较不同融合策略的性能"""
    print("\n=== 融合策略性能比较 ===")
    
    results = {}
    
    for strategy, model_path in model_paths.items():
        try:
            # 加载模型
            model, checkpoint = load_trained_model(model_path, device)
            
            # 获取测试指标
            test_metrics = checkpoint.get('test_metrics', {})
            results[strategy] = test_metrics
            
            print(f"\n{strategy} 策略:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")
                
        except Exception as e:
            print(f"加载 {strategy} 模型失败: {e}")
    
    # 找出最佳策略
    if results:
        best_strategy = max(results.keys(), key=lambda x: results[x].get('roc_auc', 0))
        print(f"\n最佳融合策略: {best_strategy} (ROC-AUC: {results[best_strategy].get('roc_auc', 0):.4f})")
    
    return results

def generate_experiment_report(fusion_strategies, used_cross_validation):
    """生成实验报告"""
    print("\n" + "="*80)
    print("实验报告")
    print("="*80)
    
    print(f"\n实验配置:")
    print(f"- 融合策略: {', '.join(fusion_strategies)}")
    print(f"- 验证方式: {'K-fold交叉验证' if used_cross_validation else '单次训练验证'}")
    print(f"- 设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    print(f"\n模型架构:")
    print(f"- 分子编码器: 3层GCN (输入维度: 6, 隐藏维度: 128)")
    print(f"- 蛋白质编码器: 3层GAT (输入维度: 20, 隐藏维度: 128)")
    print(f"- 分类器: 3层全连接网络")
    
    print(f"\n融合策略说明:")
    print(f"- concat: 简单特征拼接")
    print(f"- cross_attention: 交叉注意力机制")
    print(f"- joint_graph: 联合图融合")
    
    print(f"\n输出文件:")
    for strategy in fusion_strategies:
        if used_cross_validation:
            print(f"- fold_*_{strategy}_model.pth: {strategy}策略交叉验证模型")
        else:
            print(f"- dengue_ns5_model_{strategy}.pth: {strategy}策略完整模型")
            print(f"- best_dengue_ns5_model_{strategy}.pth: {strategy}策略最佳权重")
    print("- dengue_ns5_model_results.png: 评估结果图表")
    print("- feature_importance_analysis.png: 特征重要性分析图表")
    
    print(f"\n使用示例:")
    print(f"# 加载模型")
    print(f"model, checkpoint = load_trained_model('best_dengue_ns5_model_concat.pth')")
    print(f"# 预测新分子")
    print(f"prediction = predict_inhibitor_activity(model, new_mol_data, protein_data)")
    
    print(f"\n建议后续分析:")
    print(f"1. 比较不同融合策略的性能")
    print(f"2. 分析特征重要性结果")
    print(f"3. 进行模型解释性分析")
    print(f"4. 在独立测试集上验证模型泛化能力")
    print(f"5. 进行虚拟筛选和分子优化")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()