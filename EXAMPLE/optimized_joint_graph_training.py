import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedMolecularGraphDataset(Dataset):
    """增强的分子图数据集"""
    def __init__(self, smiles_list, labels, protein_features=None):
        self.smiles_list = smiles_list
        self.labels = labels
        self.protein_features = protein_features
        
    def smiles_to_graph(self, smiles):
        """将SMILES转换为增强的图数据"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self.create_empty_graph()
            
            # 增强的原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    atom.GetTotalNumHs(),
                    atom.GetMass(),
                    int(atom.IsInRing()),
                    atom.GetTotalValence(),
                    int(atom.GetChiralTag())
                ]
                atom_features.append(features)
            
            # 边信息和边特征
            edge_indices = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                bond_features = [
                    int(bond.GetBondType()),
                    int(bond.GetIsAromatic()),
                    int(bond.IsInRing()),
                    int(bond.GetStereo())
                ]
                
                edge_indices.extend([[i, j], [j, i]])
                edge_features.extend([bond_features, bond_features])
            
            # 分子描述符
            mol_descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCsp3(mol)
            ]
            
            if len(edge_indices) == 0:
                edge_indices = [[0, 0]]
                edge_features = [[0, 0, 0, 0]]
            
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            mol_desc = torch.tensor(mol_descriptors, dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mol_descriptors=mol_desc)
            
        except Exception as e:
            return self.create_empty_graph()
    
    def create_empty_graph(self):
        """创建空图"""
        x = torch.zeros((1, 10), dtype=torch.float)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, 4), dtype=torch.float)
        mol_desc = torch.zeros(8, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mol_descriptors=mol_desc)
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        graph = self.smiles_to_graph(smiles)
        return graph, label

class EnhancedProteinProcessor:
    """增强的蛋白质处理器"""
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.protein_graph = self.create_protein_graph()
    
    def create_protein_graph(self):
        """创建增强的蛋白质图"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', self.pdb_file)
            
            # 提取氨基酸残基特征
            residue_features = []
            residue_coords = []
            
            for residue in structure.get_residues():
                if residue.get_id()[0] == ' ':  # 只处理标准残基
                    # 残基类型编码
                    aa_dict = {'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5,
                              'GLN': 6, 'GLU': 7, 'GLY': 8, 'HIS': 9, 'ILE': 10,
                              'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14, 'PRO': 15,
                              'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20}
                    
                    resname = residue.get_resname()
                    aa_type = aa_dict.get(resname, 0)
                    
                    # 计算残基中心坐标
                    coords = []
                    for atom in residue.get_atoms():
                        coords.append(atom.get_coord())
                    
                    if coords:
                        center = np.mean(coords, axis=0)
                        residue_coords.append(center)
                        
                        # 残基特征：类型、疏水性、电荷等
                        hydrophobic = 1 if resname in ['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET'] else 0
                        charged = 1 if resname in ['ARG', 'LYS', 'ASP', 'GLU'] else 0
                        polar = 1 if resname in ['SER', 'THR', 'ASN', 'GLN', 'TYR'] else 0
                        
                        features = [aa_type, hydrophobic, charged, polar, len(coords)]
                        residue_features.append(features)
            
            if len(residue_features) == 0:
                # 创建默认蛋白质图
                x = torch.zeros((1, 5), dtype=torch.float)
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                return Data(x=x, edge_index=edge_index)
            
            # 构建蛋白质图的边（基于距离）
            residue_coords = np.array(residue_coords)
            edge_indices = []
            
            for i in range(len(residue_coords)):
                for j in range(i+1, len(residue_coords)):
                    dist = np.linalg.norm(residue_coords[i] - residue_coords[j])
                    if dist < 10.0:  # 10埃距离阈值
                        edge_indices.extend([[i, j], [j, i]])
            
            if len(edge_indices) == 0:
                edge_indices = [[0, 0]]
            
            x = torch.tensor(residue_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            return Data(x=x, edge_index=edge_index)
            
        except Exception as e:
            print(f"蛋白质处理错误: {e}")
            x = torch.zeros((1, 5), dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            return Data(x=x, edge_index=edge_index)

class OptimizedJointGraphPredictor(nn.Module):
    """优化的联合图预测器"""
    def __init__(self, mol_input_dim=10, protein_input_dim=5, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # 分子图编码器（多层GCN + GAT）
        self.mol_gcn1 = GCNConv(mol_input_dim, hidden_dim)
        self.mol_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.mol_gat1 = GATConv(hidden_dim, hidden_dim//4, heads=4, dropout=dropout)
        self.mol_gat2 = GATConv(hidden_dim, hidden_dim//4, heads=4, dropout=dropout)
        
        # 蛋白质图编码器
        self.protein_gcn1 = GCNConv(protein_input_dim, hidden_dim)
        self.protein_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.protein_gat = GATConv(hidden_dim, hidden_dim//4, heads=4, dropout=dropout)
        
        # 分子描述符处理 - 修正输入维度
        self.mol_desc_fc = nn.Sequential(
            nn.Linear(512, hidden_dim//2),  # 修正为实际输入维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # 联合图学习
        self.joint_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多尺度池化
        self.pool_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Linear(hidden_dim, hidden_dim//2)
        ])
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2 + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout//2),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, mol_batch, protein_data):
        # 分子图编码
        mol_x = F.relu(self.mol_gcn1(mol_batch.x, mol_batch.edge_index))
        mol_x = self.dropout(mol_x)
        mol_x = F.relu(self.mol_gcn2(mol_x, mol_batch.edge_index))
        mol_x = self.dropout(mol_x)
        
        mol_x_gat = F.relu(self.mol_gat1(mol_x, mol_batch.edge_index))
        mol_x_gat = self.dropout(mol_x_gat)
        mol_x_gat = F.relu(self.mol_gat2(mol_x_gat, mol_batch.edge_index))
        
        # 多尺度池化
        mol_mean = global_mean_pool(mol_x_gat, mol_batch.batch)
        mol_max = global_max_pool(mol_x_gat, mol_batch.batch)
        mol_add = global_add_pool(mol_x_gat, mol_batch.batch)
        
        # 处理分子描述符
        # 确保分子描述符的形状正确并在正确的设备上
        if mol_batch.mol_descriptors.dim() == 1:
            mol_desc_input = mol_batch.mol_descriptors.unsqueeze(0).to(mol_mean.device)
        else:
            mol_desc_input = mol_batch.mol_descriptors.to(mol_mean.device)
        
        # 如果批次中有多个分子，需要正确处理维度
        if mol_desc_input.size(0) != mol_mean.size(0):
            # 重新组织分子描述符以匹配批次大小
            batch_size = mol_mean.size(0)
            mol_desc_input = mol_desc_input.view(batch_size, -1)
        
        mol_desc = self.mol_desc_fc(mol_desc_input)
        
        # 蛋白质图编码
        protein_x = F.relu(self.protein_gcn1(protein_data.x, protein_data.edge_index))
        protein_x = self.dropout(protein_x)
        protein_x = F.relu(self.protein_gcn2(protein_x, protein_data.edge_index))
        protein_x = F.relu(self.protein_gat(protein_x, protein_data.edge_index))
        
        protein_global = global_mean_pool(protein_x, torch.zeros(protein_x.size(0), dtype=torch.long, device=protein_x.device))
        protein_global = protein_global.expand(mol_mean.size(0), -1)
        
        # 交叉注意力
        mol_attended, _ = self.cross_attention(
            mol_mean.unsqueeze(1), 
            protein_global.unsqueeze(1), 
            protein_global.unsqueeze(1)
        )
        mol_attended = mol_attended.squeeze(1)
        
        # 联合特征
        joint_features = torch.cat([mol_attended, protein_global], dim=1)
        joint_features = self.joint_transform(joint_features)
        
        # 多尺度特征融合
        pooled_mean = self.pool_layers[0](mol_mean)
        pooled_max = self.pool_layers[1](mol_max)
        
        # 最终特征组合 - 确保所有张量在同一设备上
        device = joint_features.device
        pooled_mean = pooled_mean.to(device)
        mol_desc = mol_desc.to(device)
        final_features = torch.cat([joint_features, pooled_mean, mol_desc], dim=1)
        
        # 预测
        output = self.predictor(final_features)
        return output

def load_enhanced_data():
    """加载并增强数据"""
    try:
        # 加载活性化合物数据 - 使用分号作为分隔符，处理引号
        active_df = pd.read_csv('DENV inhibitors RdRp_登革熱病毒抑制物_3739(2).csv', sep=';', quotechar='"', on_bad_lines='skip')
        active_smiles = active_df['Smiles'].dropna().tolist()
        
        # 加载非活性化合物
        inactive_df = pd.read_csv('inactive compounds_無活性抑制物(1).csv', sep=';', quotechar='"', on_bad_lines='skip')
        inactive_smiles = inactive_df['Smiles'].dropna().tolist()
        
        # 验证SMILES
        valid_active_smiles = []
        for smiles in active_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_active_smiles.append(smiles)
        
        valid_inactive_smiles = []
        for smiles in inactive_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_inactive_smiles.append(smiles)
        
        # 使用更多数据（增加到2000个活性，1000个非活性）
        active_sample_size = min(2000, len(valid_active_smiles))
        inactive_sample_size = min(1000, len(valid_inactive_smiles))
        
        np.random.seed(42)
        sampled_active = np.random.choice(valid_active_smiles, active_sample_size, replace=False)
        sampled_inactive = np.random.choice(valid_inactive_smiles, inactive_sample_size, replace=False)
        
        all_smiles = list(sampled_active) + list(sampled_inactive)
        all_labels = [1] * len(sampled_active) + [0] * len(sampled_inactive)
        
        print(f"活性化合物数量: {len(sampled_active)}")
        print(f"非活性化合物数量: {len(sampled_inactive)}")
        print(f"总化合物数量: {len(all_smiles)}")
        
        return all_smiles, all_labels
        
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None

def train_optimized_model():
    """训练优化的模型"""
    print("=== 优化Joint Graph策略训练 ===")
    
    # 加载数据
    smiles_list, labels = load_enhanced_data()
    if smiles_list is None:
        return None
    
    # 创建数据集
    dataset = EnhancedMolecularGraphDataset(smiles_list, labels)
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(smiles_list, labels)):
        print(f"\n=== 第 {fold+1} 折训练 ===")
        
        # 进一步分割训练和验证集
        train_labels = [labels[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # 创建数据加载器
        def collate_fn(batch):
            mol_graphs, labels = zip(*batch)
            return Batch.from_data_list(mol_graphs), torch.tensor(labels, dtype=torch.float)
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        # 创建模型
        model = OptimizedJointGraphPredictor(hidden_dim=512, dropout=0.2)
        protein_processor = EnhancedProteinProcessor('Dengue virus 3序列.pdb')
        protein_data = protein_processor.protein_graph
        
        # 优化器和学习率调度
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.BCELoss()
        
        # 训练循环
        best_val_auc = 0
        patience_counter = 0
        epochs = 100
        
        for epoch in range(epochs):
            # 训练
            model.train()
            total_loss = 0
            for batch_mol, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_mol, protein_data)
                loss = criterion(outputs.squeeze(), batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            # 验证
            model.eval()
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_mol, batch_labels in val_loader:
                    outputs = model(batch_mol, protein_data)
                    val_preds.extend(outputs.squeeze().tolist())
                    val_true.extend(batch_labels.tolist())
            
            val_auc = roc_auc_score(val_true, val_preds)
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f'best_optimized_model_fold_{fold}.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
            
            if patience_counter >= 20:  # 早停
                print(f"早停于第 {epoch} 轮")
                break
        
        # 加载最佳模型进行测试
        model.load_state_dict(torch.load(f'best_optimized_model_fold_{fold}.pth'))
        model.eval()
        
        test_preds = []
        test_true = []
        
        with torch.no_grad():
            for batch_mol, batch_labels in test_loader:
                outputs = model(batch_mol, protein_data)
                test_preds.extend(outputs.squeeze().tolist())
                test_true.extend(batch_labels.tolist())
        
        test_auc = roc_auc_score(test_true, test_preds)
        test_preds_binary = [1 if p > 0.5 else 0 for p in test_preds]
        test_acc = accuracy_score(test_true, test_preds_binary)
        test_f1 = f1_score(test_true, test_preds_binary)
        
        fold_result = {
            'fold': fold + 1,
            'test_auc': test_auc,
            'test_acc': test_acc,
            'test_f1': test_f1
        }
        fold_results.append(fold_result)
        
        print(f"第 {fold+1} 折结果: AUC={test_auc:.4f}, Acc={test_acc:.4f}, F1={test_f1:.4f}")
    
    # 计算平均结果
    avg_auc = np.mean([r['test_auc'] for r in fold_results])
    avg_acc = np.mean([r['test_acc'] for r in fold_results])
    avg_f1 = np.mean([r['test_f1'] for r in fold_results])
    std_auc = np.std([r['test_auc'] for r in fold_results])
    
    print(f"\n=== 5折交叉验证最终结果 ===")
    print(f"平均AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"平均准确率: {avg_acc:.4f}")
    print(f"平均F1分数: {avg_f1:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # AUC对比
    plt.subplot(2, 2, 1)
    folds = [r['fold'] for r in fold_results]
    aucs = [r['test_auc'] for r in fold_results]
    plt.bar(folds, aucs, color='skyblue', alpha=0.7)
    plt.axhline(y=0.7, color='red', linestyle='--', label='目标AUC=0.7')
    plt.axhline(y=avg_auc, color='green', linestyle='-', label=f'平均AUC={avg_auc:.3f}')
    plt.title('各折AUC结果')
    plt.xlabel('折数')
    plt.ylabel('AUC')
    plt.legend()
    plt.ylim(0.5, 1.0)
    
    # 添加数值标签
    for i, auc in enumerate(aucs):
        plt.text(i+1, auc+0.01, f'{auc:.3f}', ha='center', va='bottom')
    
    # 准确率对比
    plt.subplot(2, 2, 2)
    accs = [r['test_acc'] for r in fold_results]
    plt.bar(folds, accs, color='lightgreen', alpha=0.7)
    plt.axhline(y=avg_acc, color='green', linestyle='-', label=f'平均准确率={avg_acc:.3f}')
    plt.title('各折准确率结果')
    plt.xlabel('折数')
    plt.ylabel('准确率')
    plt.legend()
    
    # F1分数对比
    plt.subplot(2, 2, 3)
    f1s = [r['test_f1'] for r in fold_results]
    plt.bar(folds, f1s, color='orange', alpha=0.7)
    plt.axhline(y=avg_f1, color='green', linestyle='-', label=f'平均F1={avg_f1:.3f}')
    plt.title('各折F1分数结果')
    plt.xlabel('折数')
    plt.ylabel('F1分数')
    plt.legend()
    
    # 综合对比
    plt.subplot(2, 2, 4)
    metrics = ['AUC', '准确率', 'F1分数']
    values = [avg_auc, avg_acc, avg_f1]
    colors = ['skyblue', 'lightgreen', 'orange']
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('优化后Joint Graph平均性能')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('optimized_joint_graph_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fold_results, avg_auc

if __name__ == "__main__":
    results, final_auc = train_optimized_model()
    
    if final_auc >= 0.7:
        print(f"\n🎉 成功！Joint Graph策略的AUC已提升至 {final_auc:.4f}，超过了0.7的目标！")
    else:
        print(f"\n⚠️  当前AUC为 {final_auc:.4f}，仍需进一步优化以达到0.7的目标。")