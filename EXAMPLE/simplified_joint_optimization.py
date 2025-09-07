import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SimplifiedMolecularDataset:
    """简化的分子数据集"""
    def __init__(self, smiles_list, labels):
        self.smiles_list = smiles_list
        self.labels = labels
    
    def smiles_to_graph(self, smiles):
        """将SMILES转换为图数据"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self.create_empty_graph()
            
            # 原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    atom.GetMass(),
                    atom.GetTotalValence(),
                    int(atom.GetChiralTag())
                ]
                atom_features.append(features)
            
            # 边信息
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])
            
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
            
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            mol_desc = torch.tensor(mol_descriptors, dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, mol_descriptors=mol_desc)
            
        except Exception as e:
            return self.create_empty_graph()
    
    def create_empty_graph(self):
        """创建空图"""
        x = torch.zeros((1, 8), dtype=torch.float)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        mol_desc = torch.zeros(8, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, mol_descriptors=mol_desc)
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        graph = self.smiles_to_graph(smiles)
        return graph, label

class OptimizedJointGraphModel(nn.Module):
    """优化的联合图模型"""
    def __init__(self, mol_input_dim=8, hidden_dim=256, dropout=0.2):
        super().__init__()
        
        # 分子图编码器
        self.mol_gcn1 = GCNConv(mol_input_dim, hidden_dim)
        self.mol_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.mol_gat = GATConv(hidden_dim, hidden_dim//4, heads=4, dropout=dropout)
        
        # 分子描述符处理
        self.mol_desc_fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim//2)
        )
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        
        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, mol_batch):
        # 分子图编码
        mol_x = F.relu(self.mol_gcn1(mol_batch.x, mol_batch.edge_index))
        mol_x = self.dropout(mol_x)
        mol_x = F.relu(self.mol_gcn2(mol_x, mol_batch.edge_index))
        mol_x = F.relu(self.mol_gat(mol_x, mol_batch.edge_index))
        
        # 全局池化
        mol_global = global_mean_pool(mol_x, mol_batch.batch)
        
        # 处理分子描述符
        # 确保分子描述符维度正确
        batch_size = mol_global.size(0)
        if hasattr(mol_batch, 'mol_descriptors'):
            # 重新组织分子描述符以匹配批次
            mol_desc_list = []
            for i in range(batch_size):
                # 获取每个样本的分子描述符
                if mol_batch.mol_descriptors.dim() == 1:
                    mol_desc_list.append(mol_batch.mol_descriptors)
                else:
                    if i < mol_batch.mol_descriptors.size(0):
                        mol_desc_list.append(mol_batch.mol_descriptors[i])
                    else:
                        mol_desc_list.append(torch.zeros(8, device=mol_global.device))
            mol_desc_input = torch.stack(mol_desc_list)
        else:
            mol_desc_input = torch.zeros(batch_size, 8, device=mol_global.device)
        
        mol_desc_features = self.mol_desc_fc(mol_desc_input)
        
        # 特征融合
        combined_features = torch.cat([mol_global, mol_desc_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # 预测
        output = self.predictor(fused_features)
        return output.squeeze()

def load_and_prepare_data():
    """加载和准备数据"""
    try:
        # 加载数据
        active_df = pd.read_csv('DENV inhibitors RdRp_登革熱病毒抑制物_3739(2).csv', 
                               sep=';', quotechar='"', on_bad_lines='skip')
        inactive_df = pd.read_csv('inactive compounds_無活性抑制物(1).csv', 
                                 sep=';', quotechar='"', on_bad_lines='skip')
        
        # 提取SMILES
        active_smiles = active_df['Smiles'].dropna().tolist()[:2500]  # 增加数据量
        inactive_smiles = inactive_df['Smiles'].dropna().tolist()[:1500]
        
        # 创建标签
        smiles_list = active_smiles + inactive_smiles
        labels = [1] * len(active_smiles) + [0] * len(inactive_smiles)
        
        print(f"活性化合物: {len(active_smiles)}")
        print(f"非活性化合物: {len(inactive_smiles)}")
        print(f"总数据量: {len(smiles_list)}")
        
        return smiles_list, labels
        
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None

def train_optimized_model():
    """训练优化模型"""
    # 加载数据
    smiles_list, labels = load_and_prepare_data()
    if smiles_list is None:
        return None
    
    # 数据分割
    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        smiles_list, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_smiles, val_smiles, train_labels, val_labels = train_test_split(
        train_smiles, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # 创建数据集
    train_dataset = SimplifiedMolecularDataset(train_smiles, train_labels)
    val_dataset = SimplifiedMolecularDataset(val_smiles, val_labels)
    test_dataset = SimplifiedMolecularDataset(test_smiles, test_labels)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 模型和优化器
    model = OptimizedJointGraphModel(hidden_dim=512, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 训练
    best_auc = 0
    patience_counter = 0
    max_patience = 20
    
    train_losses = []
    val_aucs = []
    
    print("\n开始训练...")
    for epoch in range(150):  # 增加训练轮数
        # 训练阶段
        model.train()
        total_loss = 0
        for batch_graphs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_graphs)
            loss = criterion(outputs, batch_labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证阶段
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_graphs, batch_labels in val_loader:
                outputs = model(batch_graphs)
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_true, val_preds)
        val_aucs.append(val_auc)
        scheduler.step(val_auc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")
        
        # 早停和最佳模型保存
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_optimized_joint_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"早停于epoch {epoch}")
                break
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_optimized_joint_model.pth'))
    model.eval()
    
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for batch_graphs, batch_labels in test_loader:
            outputs = model(batch_graphs)
            test_preds.extend(outputs.cpu().numpy())
            test_true.extend(batch_labels.cpu().numpy())
    
    # 计算最终指标
    test_auc = roc_auc_score(test_true, test_preds)
    test_acc = accuracy_score(test_true, [1 if p > 0.5 else 0 for p in test_preds])
    test_f1 = f1_score(test_true, [1 if p > 0.5 else 0 for p in test_preds])
    
    print(f"\n=== 最终结果 ===")
    print(f"测试集 AUC: {test_auc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测试集 F1分数: {test_f1:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs)
    plt.title('验证集AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.axhline(y=0.7, color='r', linestyle='--', label='目标AUC=0.7')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimized_joint_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'auc': test_auc,
        'accuracy': test_acc,
        'f1': test_f1,
        'best_val_auc': best_auc
    }

if __name__ == "__main__":
    print("=== 优化Joint Graph策略训练 ===")
    results = train_optimized_model()
    
    if results:
        print(f"\n=== 训练完成 ===")
        print(f"最终AUC: {results['auc']:.4f}")
        if results['auc'] >= 0.7:
            print("🎉 成功达到AUC ≥ 0.7的目标！")
        else:
            print(f"距离目标AUC=0.7还差: {0.7 - results['auc']:.4f}")
    else:
        print("训练失败")