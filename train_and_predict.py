import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch.nn.init as init
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl

from matplotlib import rcParams
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']

# # 权重初始化函数
# def initialize_weights(model, method='kaiming'):
#     """初始化模型权重
#     Args:
#         method: 'kaiming'(默认) / 'xavier' / 'normal' / 'zeros'
#     """
#     for m in model.modules():
#         if isinstance(m, nn.Linear):
#             if method == 'kaiming':
#                 init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             elif method == 'xavier':
#                 init.xavier_normal_(m.weight)
#             elif method == 'normal':
#                 init.normal_(m.weight, mean=0, std=0.01)
#             elif method == 'zeros':
#                 init.zeros_(m.weight)
#             else:
#                 raise ValueError(f"Unknown init method: {method}")
            
#             if m.bias is not None:
#                 init.zeros_(m.bias)

#强制字体
def force_times_new_roman(ax):
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
        # label.set_fontsize(fontsize_tick)

    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        # label.set_fontsize(fontsize_tick)
# 自定义数据集类
class MultiTaskDataset(Dataset):
    def __init__(self, X, y_reg, y_cls):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], (self.y_reg[idx], self.y_cls[idx])

# 多任务模型
class MTLModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MTLModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
        )
        self.reg_output = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.cls_output = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(),
            nn.Linear(32, 1 if num_classes == 2 else num_classes),
            nn.Softmax(dim=1) if num_classes > 2 else nn.Sigmoid()
        )
        
        # 初始化权重
        # initialize_weights(self, method=init_method)
    
    def forward(self, x):
        x = self.shared(x)
        reg_out = self.reg_output(x)
        cls_out = self.cls_output(x)
        return reg_out, cls_out

# 数据准备
def load_data(file_path):
    df = pd.read_excel(file_path).dropna()
    df_shuffled = df.sample(frac=1, random_state=41).reset_index(drop=True)
    X = df_shuffled.iloc[:, 4:].values
    y_reg = df_shuffled.iloc[:, 2].values.astype(np.float32)
    y_cls = LabelEncoder().fit_transform(df_shuffled.iloc[:, 3])
    
    # X = df.iloc[:, 4:].values
    # y_reg = df.iloc[:, 2].values.astype(np.float32)
    # y_cls = LabelEncoder().fit_transform(df.iloc[:, 3])

    X_test = X[:16]
    y_reg_test = y_reg[:16]
    y_cls_test = y_cls[:16]
    X = X[16:]
    y_reg = y_reg[16:]
    y_cls = y_cls[16:]
    return X, y_reg, y_cls, X_test, y_reg_test ,y_cls_test 

#回归数据增强
class RegressionAugmentation:
    def __init__(self, noise_std=0.05):
        self.noise_std = noise_std

    def __call__(self, batch):
        xs = []
        y_regs = []
        y_clss = []

        for x, (y_reg, y_cls) in batch:
            xs.append(x)
            y_regs.append(y_reg)
            y_clss.append(y_cls)

        xs = torch.stack(xs)
        y_regs = torch.stack(y_regs).float()
        y_clss = torch.stack(y_clss).long()

        # 回归标签加噪声增强
        noise = torch.randn_like(y_regs) * self.noise_std
        y_regs_aug = y_regs + noise

        return xs, (y_regs_aug, y_clss)

class MultiTaskMixup:
    def __init__(self, alpha=0.2, soft_label=False):
        self.alpha = alpha
        self.soft_label = soft_label

    def __call__(self, batch):
        xs = []
        y_regs = []
        y_clss = []

        # ---- 1. 解包 batch ----
        for x, (y_reg, y_cls) in batch:
            xs.append(x)
            y_regs.append(y_reg)
            y_clss.append(y_cls)

        xs = torch.stack(xs)                    # (B, ...)
        y_regs = torch.stack(y_regs).float()    # (B,)
        y_clss = torch.stack(y_clss).long()     # (B,)

        # ---- 2. Mixup 系数 ----
        lam = np.random.beta(self.alpha, self.alpha)

        # ---- 3. 随机打乱 batch ----
        idx = torch.randperm(xs.size(0))

        # ---- 4. Mixup 特征 ----
        xs_mix = lam * xs + (1 - lam) * xs[idx]

        # ---- 5. Mixup 回归标签（线性混合）----
        y_regs_mix = lam * y_regs + (1 - lam) * y_regs[idx]

        # ---- 6. 分类标签的 Mixup ----
        if self.soft_label:
            # 转换成 one-hot
            num_classes = torch.max(y_clss).item() + 1
            y1 = torch.nn.functional.one_hot(y_clss, num_classes=num_classes).float()
            y2 = torch.nn.functional.one_hot(y_clss[idx], num_classes=num_classes).float()
            y_clss_mix = lam * y1 + (1 - lam) * y2
        else:
            # 不做混合，保持硬标签
            y_clss_mix = y_clss

        # ---- 7. 返回结构必须与原始 Dataset 一致 ----
        return xs_mix, (y_regs_mix, y_clss_mix)

def plot_vertical_shap_like_reference(shap_values, features, feature_names, filename='shap_vertical_combined.png'):
    """精确复制参考图片的垂直排版样式"""

    # mpl.rcParams['font.family'] = font_prop.get_name()
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # 创建垂直排列的子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # ===== 上图：条形图（Mean Shapley Value） =====
    plt.sca(ax1)
    shap.summary_plot(
        shap_values, 
        features, 
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=15
    )
    force_times_new_roman(ax1)
    
    # 自定义条形图样式
    ax1.set_title("Mean Shapley Value (Feature Importance)", 
                 fontsize=16, fontweight='bold', pad=20, fontname= 'Times New Roman')
    ax1.set_xlabel("Shapley Value Contribution (Bee Swarm)", fontsize=12, fontweight='bold', fontname= 'Times New Roman')
    
    # 美化条形图
    for container in ax1.containers:
        for bar in container:
            bar.set_color('#2E86AB')  # 蓝色调
            bar.set_alpha(0.8)
            bar.set_edgecolor('#1C5D7A')
    
    # 设置网格线
    ax1.grid(False, axis='x', alpha=0.3, linestyle='-', color='#cccccc')
    ax1.set_axisbelow(False)
    
    # ===== 下图：蜂群图（Bee Swarm） =====
    plt.sca(ax2)
    shap.summary_plot(
        shap_values, 
        features, 
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=15,
        color_bar=False
    )
    
    # 自定义蜂群图标题
    ax2.set_title("Shapley Value Contribution (Bee Swarm)", 
                 fontsize=16, fontweight='bold', pad=20, fontname= 'Times New Roman')
    ax2.set_xlabel("SHAP value impact on model output", fontsize=12, fontname= 'Times New Roman')
    
    # 添加颜色条说明（手动添加文本）
    # ax2.text(0.02, -0.15, "High", transform=ax2.transAxes, 
    #          fontsize=10, color='red', 
    #          fontweight='bold',
    #          fontname= 'Times New Roman')
    # ax2.text(0.95, -0.15, "Low", transform=ax2.transAxes, 
    #          fontsize=10, color='blue', 
    #          fontweight='bold',
    #          ha='right', fontname= 'Times New Roman')
    # ax2.text(0.5, -0.15, "Feature value", transform=ax2.transAxes,
    #          fontsize=10, ha='center', 
    #          fontweight='bold', 
    #          fontname= 'Times New Roman')
    
    force_times_new_roman(ax2)
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(filename, dpi=600, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"垂直排列SHAP图已保存: {filename}")

def plot_transparent_overlay_shap(shap_values, features, feature_names, filename='shap_transparent_overlay.png'):
    """半透明重叠效果"""
    
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # ax2 = ax.twiny() 
    
    
    
    
    
    # ===== 再绘制蜂群图 =====
    # 使用相同的坐标轴绘制蜂群图
    shap.summary_plot(
        shap_values, 
        features, 
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=10
    )
    
    force_times_new_roman(ax)

    # swarm_xmin, swarm_xmax = ax.get_xlim()

    # ===== 先绘制条形图 =====
    # 计算平均SHAP值
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]
    # sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_idx_10 = sorted_idx[:10]#取前十
    sorted_features = [feature_names[i] for i in sorted_idx_10]
    sorted_mean_shap = mean_shap[sorted_idx_10[::-1]]
    
    # 绘制浅蓝色条形图（半透明）
    y_pos = np.arange(len(sorted_features))
    bars = ax.barh(y_pos, sorted_mean_shap, 
                   color='#ADD8E6', alpha=0.4,  # 更浅的蓝色，更透明
                   edgecolor='#87CEEB', linewidth=1,
                   label='Mean |SHAP value|')
    
   

    # ax2.set_xlim(0, 0.5) 
    # ax2.set_xlabel("Mean Shapley Value (Feature Importance)", fontsize=12)
    # ax2.tick_params(axis='x', labelcolor='#4682B4', labeltop=True)  # 标签在上方
    
    

    # 调整ax1的范围，为条形图留出空间
    # ax.set_xlim(-max_bar * 1.2, swarm_xmax)
    # 设置标题和标签
    ax.set_title("Mean Shapley Value (Feature Importance)", 
                 fontsize=12,fontname = 'Times New Roman')
    ax.set_xlabel("Shapley Value Contribution (Bee Swarm)", fontsize=12,fontname = 'Times New Roman')
    
    # 添加图例
    ax.legend(loc='lower right', frameon=True, facecolor='white',prop={'family': 'Times New Roman', 'size': 12})
    
    # 美化网格线
    ax.grid(False, axis='x', alpha=0.2, linestyle='-', color='#999999')
    ax.set_axisbelow(False)

    # ax2.xaxis.set_label_position('top')  # x轴标签在上方
    # ax2.xaxis.set_ticks_position('top')  # x轴标签在上方

    for spine in fig.gca().spines.values():
        spine.set_linewidth(2)  # 增加边框线宽
        spine.set_color('black')  # 设置边框颜色
    
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"半透明重叠SHAP图已保存: {filename}")

#shap处理
def perform_shap_analysis(model, X_train, X_val, num_classes, device,feature_names_ori):
    model.eval()
    # 准备背景数据和测试数据
    background = torch.tensor(X_train[:], dtype=torch.float32).to(device)  # 使用前100个样本作为背景
    test_samples = torch.tensor(X_val[:], dtype=torch.float32).to(device)   # 分析前50个测试样本
    
    # 定义模型输出函数
    class WrappedModel(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model
        def forward(self, x):
            return self.original_model(x)[0]
    # def model_forward():
    #     return model[0]  # 只返回回归输出
    
    # 创建解释器
    print("\n=== 进行回归SHAP分析 ===")
    reg_model_forward = WrappedModel(model)
    explainer = shap.DeepExplainer(reg_model_forward, background)
    
    # 计算SHAP值
    reg_shap_values = explainer.shap_values(test_samples)

    #print(reg_shap_values)
    # if isinstance(reg_shap_values, list):
    #     shap_values = shap_values[0]
    
    # 获取特征名称
    feature_names = feature_names_ori
    # print(feature_names)
    #feature_names = np.array([f"Feature_{i}" for i in range(X_train.shape[1])])
    # 可视化SHAP值
    
   
    # shap.summary_plot(
    #     reg_shap_values,
    #     test_samples.cpu().numpy(),
    #     feature_names=feature_names,
    #     plot_type="bar",
    #     max_display=len(feature_names),  # 显示所有特征
    #     show=True
    # )
    plot_transparent_overlay_shap(reg_shap_values.squeeze(-1),
                                        test_samples.cpu().numpy(), 
                                        feature_names, 
                                        filename='shap_vertical_combined.png')
    # shap.summary_plot(reg_shap_values.squeeze(-1) ,
    #                      test_samples.cpu().numpy(),
    #                      feature_names=feature_names,
    #                      plot_type="bar",
    #                      show=False,
    #                      max_display=15)
    # plt.tight_layout()
    # plt.savefig('shap_reg_plot.png', dpi=600, bbox_inches='tight')
    # plt.close()
    
    # 如果需要分类任务的SHAP分析，可以创建另一个解释器
    class WrappedClassifier(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model

        def forward(self, x):
            return self.original_model(x)[1]
    
    model_cls_forward = WrappedClassifier(model)
    
    
    
    if num_classes == 2:
        # def model_cls_forward():
        #     return model[1]  # 返回分类输出
        print("\n=== 进行分类SHAP分析 ===")
        cls_explainer = shap.GradientExplainer(model_cls_forward, background)
        cls_shap_values = cls_explainer.shap_values(test_samples)
        print(cls_shap_values.shape)
        shap.summary_plot(cls_shap_values, test_samples.cpu().numpy(), feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_cls_plot.png', dpi=600, bbox_inches='tight')
        plt.close()
# 自定义R2计算
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def save_to_excel(results, filename='cv_results.xlsx'):
    """保存交叉验证结果到Excel"""
    with pd.ExcelWriter(filename) as writer:
        # 保存每一折的详细结果
        pd.DataFrame(results['fold_details']).to_excel(writer, sheet_name='每折结果', index=False)
        # 保存平均结果
        pd.DataFrame([results['mean_metrics']]).to_excel(writer, sheet_name='平均结果', index=False)
        # 保存标准差
        pd.DataFrame([results['std_metrics']]).to_excel(writer, sheet_name='标准差', index=False)

#计算拟合数据
def fit_calculate(model, val_loader, test_loader, num_classes,fold, device='cpu',*results):
    model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
    model.eval()
    
    # 测试集预测结果
    all_reg_preds, all_reg_true = [], []
    all_cls_preds, all_cls_true = [], []
    
    
    with torch.no_grad():
        for inputs, (reg_targets, cls_targets) in test_loader:
            inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)
            reg_pred, cls_pred = model(inputs)
            
            all_reg_preds.extend(reg_pred.cpu().numpy().flatten())
            all_reg_true.extend(reg_targets.cpu().numpy().flatten())
            
            if num_classes == 2:
                cls_pred = (cls_pred >= 0.5).float()
            else:
                cls_pred = torch.argmax(cls_pred, dim=1)
            
            all_cls_preds.extend(cls_pred.cpu().numpy())
            all_cls_true.extend(cls_targets.cpu().numpy())
            
    ########################################################        
    # 训练集预测结果
    train_reg_preds, train_reg_true = [], []
    train_cls_preds, train_cls_true = [], []
    model.eval()
    with torch.no_grad():
        for inputs, (reg_targets, cls_targets) in val_loader:
            inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)
            reg_pred, cls_pred = model(inputs)
            
            train_reg_preds.extend(reg_pred.cpu().numpy().flatten())
            train_reg_true.extend(reg_targets.cpu().numpy().flatten())
            
            if num_classes == 2:
                cls_pred = (cls_pred >= 0.5).float()
            else:
                cls_pred = torch.argmax(cls_pred, dim=1)
            
            train_cls_preds.extend(cls_pred.cpu().numpy())
            train_cls_true.extend(cls_targets.cpu().numpy())
    
    fit_results = {
        'train_reg_preds' : train_reg_preds,
        'train_reg_true'  : train_reg_true,
        'train_cls_preds' : train_cls_preds,
        'train_cls_true' : train_cls_true,
        'test_reg_preds' : all_reg_preds,
        'test_reg_true'  : all_reg_true,
        'test_cls_preds' : all_cls_preds,
        'test_cls_true' : all_cls_true
        
    }
    
   
    with pd.ExcelWriter(f'fit_results_fold{fold}.xlsx') as writer:
        # 保存训练集和测试集结果到不同Sheet
        pd.DataFrame({
        'train_reg_preds': fit_results['train_reg_preds'],
        'train_reg_true': fit_results['train_reg_true'],
        'train_cls_preds': fit_results['train_cls_preds'],
        'train_cls_true': fit_results['train_cls_true']
        }).to_excel(writer, sheet_name='训练集结果', index=False)
        
        pd.DataFrame(
            {
            'test_reg_preds': fit_results['test_reg_preds'],
            'test_reg_true': fit_results['test_reg_true'],
            'test_cls_preds': fit_results['test_cls_preds'],
            'test_cls_true': fit_results['test_cls_true'],
        }).to_excel(writer, sheet_name='测试集结果', index=False)
    ############################################################# 

# 训练和验证函数
def train_and_validate(model, train_loader, val_loader, num_classes,fold = 0, device='cpu'):
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)#nn.BCELoss() if num_classes == 2 else nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.001,
    amsgrad=False)
    
    best_val_loss = float('inf')
    patience = 2000
    patience_counter = 0
    
    for epoch in range(4000):  # 每折训练5000个epoch（数据量太少）
        model.train()
        train_loss = 0.0
        for inputs, (reg_targets, cls_targets) in train_loader:
            inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)
            optimizer.zero_grad()
            reg_pred, cls_pred = model(inputs)
            
            reg_loss = reg_criterion(reg_pred.squeeze(), reg_targets)
            if num_classes == 2:
                cls_loss = cls_criterion(cls_pred.squeeze(), cls_targets.float())
            else:
                cls_loss = cls_criterion(cls_pred, cls_targets)
            
            loss = 0.6 * reg_loss + 0.4 * cls_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, (reg_targets, cls_targets) in val_loader:
                inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)
                reg_pred, cls_pred = model(inputs)
                
                reg_loss = reg_criterion(reg_pred.squeeze(), reg_targets)
                
                if num_classes == 2:
                    cls_loss = cls_criterion(cls_pred.squeeze(), cls_targets.float())
                else:
                    cls_loss = cls_criterion(cls_pred, cls_targets)
                
                val_loss += (0.6 * reg_loss + 0.4 * cls_loss).item()
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # 加载最佳模型进行验证集评估
    model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
    model.eval()
    
    all_reg_preds, all_reg_true = [], []
    all_cls_preds, all_cls_true = [], []
    
    with torch.no_grad():
        for inputs, (reg_targets, cls_targets) in val_loader:
            inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)
            reg_pred, cls_pred = model(inputs)
            
            all_reg_preds.extend(reg_pred.cpu().numpy().flatten())
            all_reg_true.extend(reg_targets.cpu().numpy().flatten())
            
            if num_classes == 2:
                cls_pred = (cls_pred >= 0.5).float()
            else:
                cls_pred = torch.argmax(cls_pred, dim=1)
            
            all_cls_preds.extend(cls_pred.cpu().numpy())
            all_cls_true.extend(cls_targets.cpu().numpy())
    
    # 计算指标
    reg_metrics = {
        'R2': r2_score(all_reg_true, all_reg_preds),
        'MAE': mean_absolute_error(all_reg_true, all_reg_preds),
        'RMSE': np.sqrt(mean_squared_error(all_reg_true, all_reg_preds))
    }
    
    if num_classes == 2:
        auc = roc_auc_score(all_cls_true, all_cls_preds)
    else:
        auc = roc_auc_score(all_cls_true, all_cls_preds, multi_class='ovr')
    
    cls_metrics = {
        'Accuracy': accuracy_score(all_cls_true, all_cls_preds),
        'Precision': precision_score(all_cls_true, all_cls_preds, average='macro'),
        'Recall': recall_score(all_cls_true, all_cls_preds, average='macro'),
        'F1': f1_score(all_cls_true, all_cls_preds, average='macro'),
        'AUC': auc
    }
    
    return {**reg_metrics, **cls_metrics}

# def train_and_test(model,final_train_loader,test_loader,num_classes,fold,device='cpu'):
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)#nn.BCELoss() if num_classes == 2 else nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.001,
    amsgrad=False)
    
    best_val_loss = float('inf')
    patience = 2000
    patience_counter = 0
    
    for epoch in range(4000):  # 每折训练5000个epoch（数据量太少）
        model.train()
        train_loss = 0.0
        for inputs, (reg_targets, cls_targets) in final_train_loader:
            inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)
            optimizer.zero_grad()
            reg_pred, cls_pred = model(inputs)
            
            
            reg_loss = reg_criterion(reg_pred.squeeze(), reg_targets)
            if num_classes == 2:
                cls_loss = cls_criterion(cls_pred.squeeze(), cls_targets.float())
            else:
                cls_loss = cls_criterion(cls_pred, cls_targets)
            
            loss = 0.6 * reg_loss + 0.4 * cls_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, (reg_targets, cls_targets) in test_loader:
                inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)
                reg_pred, cls_pred = model(inputs)
                
                
                reg_loss = reg_criterion(reg_pred.squeeze(), reg_targets)
                if num_classes == 2:
                    cls_loss = cls_criterion(cls_pred.squeeze(), cls_targets.float())
                else:
                    cls_loss = cls_criterion(cls_pred, cls_targets)
                
                val_loss += (0.6 * reg_loss + 0.4 * cls_loss).item()
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # 加载最佳模型进行验证集评估
    model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
    model.eval()
    
    all_reg_preds, all_reg_true = [], []
    all_cls_preds, all_cls_true = [], []
    
    with torch.no_grad():
        for inputs, (reg_targets, cls_targets) in test_loader:
            inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)
            reg_pred, cls_pred = model(inputs)
            
            all_reg_preds.extend(reg_pred.cpu().numpy().flatten())
            all_reg_true.extend(reg_targets.cpu().numpy().flatten())
            
            if num_classes == 2:
                cls_pred = (cls_pred >= 0.5).float()
            else:
                cls_pred = torch.argmax(cls_pred, dim=1)
            
            all_cls_preds.extend(cls_pred.cpu().numpy())
            all_cls_true.extend(cls_targets.cpu().numpy())
    
    # 计算指标
    reg_metrics = {
        'R2': r2_score(all_reg_true, all_reg_preds),
        'MAE': mean_absolute_error(all_reg_true, all_reg_preds),
        'RMSE': np.sqrt(mean_squared_error(all_reg_true, all_reg_preds))
    }
    
    if num_classes == 2:
        auc = roc_auc_score(all_cls_true, all_cls_preds)
    else:
        auc = roc_auc_score(all_cls_true, all_cls_preds, multi_class='ovr')
    
    cls_metrics = {
        'Accuracy': accuracy_score(all_cls_true, all_cls_preds),
        'Precision': precision_score(all_cls_true, all_cls_preds, average='macro'),
        'Recall': recall_score(all_cls_true, all_cls_preds, average='macro'),
        'F1': f1_score(all_cls_true, all_cls_preds, average='macro'),
        'AUC': auc
    }
    
    return {**reg_metrics, **cls_metrics}

# 主流程
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    #加载数据
    X, y_reg, y_cls, X_test, y_reg_test, y_cls_test = load_data('LYC吸附能+灵敏度.xlsx')

    
    feature_names_ori = np.array(pd.read_excel('LYC吸附能+灵敏度-英文.xlsx', index_col=0).columns[3:])

    num_classes = len(np.unique(y_cls))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.fit_transform(X_test)
    final_train_dataset = MultiTaskDataset(X,y_reg,y_cls)
    test_dataset = MultiTaskDataset(X_test, y_reg_test, y_cls_test)

    final_train_loader = DataLoader(final_train_dataset, batch_size=32, shuffle=True,collate_fn=MultiTaskMixup())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储结果
    results = {
        'fold_details': [],
        'mean_metrics': defaultdict(float),
        'std_metrics': defaultdict(float)
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1}/5 ===")
        
        # 数据划分
        X_train, X_val = X[train_idx], X[val_idx]
        y_reg_train, y_reg_val = y_reg[train_idx], y_reg[val_idx]
        y_cls_train, y_cls_val = y_cls[train_idx], y_cls[val_idx]
        
        # 标准化
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_val = scaler.transform(X_val)
        
        # 数据加载器

        train_dataset = MultiTaskDataset(X_train, y_reg_train, y_cls_train)
        val_dataset = MultiTaskDataset(X_val, y_reg_val, y_cls_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=MultiTaskMixup())
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        
        # 模型初始化
        model = MTLModel(X.shape[1], num_classes).to(device)
        
        # 训练和验证
        fold_metrics = train_and_validate(model, train_loader, val_loader, num_classes, fold + 1, device)
        results['fold_details'].append({'Fold': fold + 1, **fold_metrics})
        
        # 打印当前折结果
        print(f"Fold {fold + 1} \n回归 R²: {fold_metrics['R2']:.4f},MAE:{fold_metrics['MAE']:.4f},RMSE:{fold_metrics['RMSE']:.4f};\n分类 Accuracy: {fold_metrics['Accuracy']:.4f},Precision: {fold_metrics['Precision']:.4f},Recall: {fold_metrics['Recall']:.4f},F1: {fold_metrics['F1']:.4f},AUC: {fold_metrics['AUC']:.4f}")
    
    
    
    # 计算平均结果和标准差
    df_folds = pd.DataFrame(results['fold_details'])
    for metric in df_folds.columns[1:]:
        results['mean_metrics'][f'Mean_{metric}'] = df_folds[metric].mean()
        results['std_metrics'][f'Std_{metric}'] = df_folds[metric].std()
    
    # 保存结果
    save_to_excel(results)
    print("\n=== 5折交叉验证平均结果 ===")
    for k, v in results['mean_metrics'].items():
        print(f"{k}: {v:.4f}")
    print("\n=== 标准差 ===")
    for k, v in results['std_metrics'].items():
        print(f"{k}: {v:.4f}")
    print("\n模型性能（均值 ± 标准差）：")
    i = 0
    for k, v in results['mean_metrics'].items():
        print(f"{k}:{v:.4f} ± {list(results['std_metrics'].values())[i]:.4f}")
        i += 1
        
    #计算测试结果   
    test_metrics = train_and_validate(model,final_train_loader,test_loader,num_classes,0,device)
    results['fold_details'].append({'Fold': 0, **test_metrics})
    print(f"测试结果 \n回归 R²: {fold_metrics['R2']:.4f},MAE:{fold_metrics['MAE']:.4f},RMSE:{fold_metrics['RMSE']:.4f};\n分类 Accuracy: {fold_metrics['Accuracy']:.4f},Precision: {fold_metrics['Precision']:.4f},Recall: {fold_metrics['Recall']:.4f},F1: {fold_metrics['F1']:.4f},AUC: {fold_metrics['AUC']:.4f}")
    
    #########保存指定者的验证和测试结果
    #fit_calculate(model, final_train_loader, test_loader, num_classes, 0, device,results['fold_details'])

    # SHAP分析
    print("\n=== 进行SHAP分析 ===")
    perform_shap_analysis(model, X, X_test, num_classes, device,feature_names_ori)

if __name__ == "__main__":
    main()