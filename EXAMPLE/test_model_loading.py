import torch
import os

def test_model_loading():
    """测试模型文件加载"""
    model_files = [
        'best_dengue_ns5_model_concat.pth',
        'best_dengue_ns5_model_cross_attention.pth', 
        'best_dengue_ns5_model_joint_graph.pth'
    ]
    
    print("=== 测试模型文件加载 ===")
    
    for model_file in model_files:
        print(f"\n检查模型文件: {model_file}")
        
        # 检查文件是否存在
        if not os.path.exists(model_file):
            print(f"❌ 文件不存在: {model_file}")
            continue
        
        # 检查文件大小
        file_size = os.path.getsize(model_file)
        print(f"✅ 文件存在，大小: {file_size / (1024*1024):.2f} MB")
        
        # 尝试加载模型
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            print(f"✅ 模型加载成功")
            
            # 检查模型结构
            if isinstance(checkpoint, dict):
                print(f"  模型是字典格式，包含键: {list(checkpoint.keys())}")
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  模型参数数量: {len(state_dict)}")
                    print(f"  前5个参数键: {list(state_dict.keys())[:5]}")
            else:
                print(f"  模型是直接状态字典格式")
                print(f"  参数数量: {len(checkpoint)}")
                print(f"  前5个参数键: {list(checkpoint.keys())[:5]}")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_model_loading()