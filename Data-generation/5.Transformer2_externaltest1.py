import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
import csv
from datetime import datetime
import gc # 导入垃圾回收模块

# ---------------------------------------------------------
# 模型定义：Tabular Transformer
# ---------------------------------------------------------
class TabularTransformer(nn.Module):
    def __init__(self, input_dim=10, d_model=32, nhead=4, num_layers=1, dropout=0.3):
        super(TabularTransformer, self).__init__()
        self.feature_projection = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            dropout=dropout,
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model * input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_projection(x)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)

# ---------------------------------------------------------
# 批次化预测函数：防止大文件预测时显存崩溃 (核心优化点)
# ---------------------------------------------------------
def predict_in_batches(model, X_data, device, batch_size=1024):
    model.eval()
    preds = []
    # 使用较宽的 batch 执行推理，既快又省显存
    with torch.no_grad():
        for i in range(0, len(X_data), batch_size):
            batch_x = torch.from_numpy(X_data[i:i+batch_size]).to(device)
            batch_pred = model(batch_x).cpu().numpy()
            preds.append(batch_pred)
    return np.vstack(preds)

# ---------------------------------------------------------
# 数据预处理
# ---------------------------------------------------------
def load_and_scale(file_path, scaler_x=None, scaler_y=None):
    df = pd.read_csv(file_path)
    x_cols = [f'x{i}' for i in range(1, 11)]
    X = df[x_cols].values.astype(np.float32)
    y = df['y'].values.astype(np.float32).reshape(-1, 1)
    
    if scaler_x is None:
        scaler_x = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y)
        
    return scaler_x.transform(X), scaler_y.transform(y), scaler_x, scaler_y

# ---------------------------------------------------------
# 核心训练逻辑
# ---------------------------------------------------------
def train_and_evaluate(train_file, ext_file, device):
    # 1. 加载数据
    X_raw, y_raw, sx, sy = load_and_scale(train_file)
    data_count = len(X_raw)
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # 调大 batch_size 可加快 3090 训练速度，这里保持 16 以兼容小样本
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True) 
    
    model = TabularTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    criterion = nn.MSELoss()
    
    # 2. 训练
    best_val_r2 = -np.inf
    best_state = None
    
    for epoch in range(300):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
        # 每个 Epoch 评估 (使用小规模验证集)
        model.eval()
        with torch.no_grad():
            v_input = torch.from_numpy(X_val).to(device)
            v_pred = model(v_input).cpu().numpy()
            v_r2 = r2_score(sy.inverse_transform(y_val), sy.inverse_transform(v_pred))
            if v_r2 > best_val_r2:
                best_val_r2 = v_r2
                best_state = {k: v.cpu() for k, v in model.state_dict().items()} # 存入 CPU 显存
    
    # 3. 最终评估 (使用批次化预测，防止全量数据撑爆显存)
    model.load_state_dict(best_state)
    t_pred = predict_in_batches(model, X_train, device)
    train_r2 = r2_score(sy.inverse_transform(y_train), sy.inverse_transform(t_pred))
    
    X_ext_raw, y_ext_raw, _, _ = load_and_scale(ext_file, sx, sy)
    e_pred = predict_in_batches(model, X_ext_raw, device)
    ext_r2 = r2_score(sy.inverse_transform(y_ext_raw), sy.inverse_transform(e_pred))
    
    print(f"Metrics: Train={train_r2:.4f} | Val={best_val_r2:.4f} | Test={ext_r2:.4f}")
    
    # 返回结果并清理局部大变量
    res = (data_count, train_r2, best_val_r2, ext_r2)
    del model, optimizer, train_loader, X_train, X_val
    return res

# ---------------------------------------------------------
# 主程序
# ---------------------------------------------------------
import time
if __name__ == "__main__":
    FUN_NUM = 4      
    EXTEST_FILE = f"external_test_F{FUN_NUM}.csv"
    INPUT_FOLDER = f"F{FUN_NUM}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%H%M%S_%f")
    SUMMARY_FILE = f"summary_F{FUN_NUM}_{timestamp}.csv"
    
    csv_files = sorted(list(Path(INPUT_FOLDER).glob("*.csv")))
    
    # 初始化表格
    with open(SUMMARY_FILE, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["Filename", "Data_Size", "Train_R2", "Validation_R2", "Test_R2"])

    for f in csv_files:
        print(f"\n>>> Processing: {f.name}")
        start_time = time.time()  # 记录程序开始时间
        try:
            count, tr_r2, va_r2, te_r2 = train_and_evaluate(str(f), EXTEST_FILE, device)
            
            with open(SUMMARY_FILE, mode='a', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([f.name, count, f"{tr_r2:.4f}", f"{va_r2:.4f}", f"{te_r2:.4f}"])
            
            # 每跑完一个文件，强制进行垃圾回收核显存清理
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error at {f.name}: {e}")
            torch.cuda.empty_cache()
        
        end_time = time.time()  # 记录程序结束时间
        elapsed_time = end_time - start_time  # 计算总耗时
        print(f"\n程序运行总耗时: {elapsed_time:.2f} 秒")

    print("\nAll tasks completed.")