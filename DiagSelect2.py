import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# 1. 智能体模型 (Section III-B, Eq. 6-11)
# ==========================================
class DiagSelectAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim=80):  # 论文 Section IV-A-1: H=80
        super(DiagSelectAgent, self).__init__()
        # 对应 Eq. (6-9): GRU 提取样本间的时序依赖或全局统计特征
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # 对应 Eq. (10): 线性层映射
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, h_prev):
        # x: [1, num_samples, feat_dim + num_classes]
        out, h_next = self.gru(x, h_prev)
        # 对应 Eq. (11): Softmax 得到选择/不选的概率分布
        logits = self.fc(out)
        probs = F.softmax(logits, dim=-1)
        return probs.squeeze(0), h_next


# ==========================================
# 2. 基础诊断模型 (Section III-A, f_dig)
# ==========================================
class DiagnosisMLP(nn.Module):
    """
    一个更真实的诊断模型，增加深度以符合实际工业场景
    """

    def __init__(self, input_dim, num_classes):
        super(DiagnosisMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. 核心算法类 (Algorithm 1 & Section III-D)
# ==========================================
class DiagSelectFramework:
    def __init__(self, feat_dim, num_classes, agent_lr=1e-3, base_lr=1e-2):
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.agent = DiagSelectAgent(input_dim=feat_dim + num_classes)

        # 论文 Section IV-A-1: Agent 使用 RMSprop 优化器
        self.agent_optimizer = optim.RMSprop(self.agent.parameters(), lr=agent_lr)

    def _calculate_g_mean(self, y_true, y_pred):
        """
        对应 Section III-D: 评价指标 G-Mean
        """
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        with np.errstate(divide='ignore', invalid='ignore'):
            recalls = np.diag(cm) / cm.sum(axis=1)
            recalls = np.nan_to_num(recalls)
        return np.exp(np.mean(np.log(recalls + 1e-7)))

    def train_base_model(self, train_loader, val_x, val_y):
        """
        真实的诊断模型训练过程，包含完整的 Epoch 迭代
        """
        model = DiagnosisMLP(self.feat_dim, self.num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.01)  # 基础模型常用 Adam
        criterion = nn.CrossEntropyLoss()

        # 更加真实的训练周期 (不再是之前的 10 step)
        model.train()
        for epoch in range(30):
            for batch_x, batch_y in train_loader:
                if batch_x.size(0) <= 1:
                    continue
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # 验证集评估
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(val_x), dim=1).cpu().numpy()
            y_true = val_y.cpu().numpy()
            # 论文关键：使用 Macro-F1 作为主要奖励信号 (Eq. 14)
            f1 = f1_score(y_true, preds, average='macro')
            gmean = self._calculate_g_mean(y_true, preds)
            # 综合奖励 (可根据论文具体偏好调整比例)
            reward = 0.5 * f1 + 0.5 * gmean
        return reward

    def run_offline_training(self, train_x, train_y, val_x, val_y, T=10, epi=5):
        """
        严格执行 Algorithm 1: Offline Training
        """
        # Algorithm 1, Line 3: 初始化隐状态
        h_t = torch.zeros(1, 1, self.agent.hidden_dim)

        for t in range(T):
            # Algorithm 1, Line 6: 构造状态 s_t (特征+标签)
            y_onehot = F.one_hot(train_y, num_classes=self.num_classes).float()
            s_t = torch.cat([train_x, y_onehot], dim=1).unsqueeze(0)

            # Algorithm 1, Line 7: 获取选择概率
            # 此时计算图开始记录
            probs, h_t = self.agent(s_t, h_t)

            trajectories = []  # 记录 epi 次采样的结果

            for k in range(epi):
                # Eq. (12): 采样动作
                m = torch.distributions.Categorical(probs)
                actions = m.sample()
                log_probs = m.log_prob(actions).sum()

                # Algorithm 1, Line 10: 筛选子集
                selected_mask = (actions == 1)
                if selected_mask.sum() < self.num_classes * 2:  # 保护机制：样本太少无法训练
                    reward = 0.0
                else:
                    sel_x = train_x[selected_mask]
                    sel_y = train_y[selected_mask]

                    # 封装成 DataLoader 进行标准训练
                    ds = TensorDataset(sel_x, sel_y)
                    loader = DataLoader(ds, batch_size=32, shuffle=True)
                    reward = self.train_base_model(loader, val_x, val_y)

                trajectories.append((log_probs, reward))

            # --- 策略梯度更新 (Eq. 15) ---
            # 计算 epi 次采样的平均奖励作为 Baseline 以减小方差
            rewards = [tr[1] for tr in trajectories]
            mean_reward = np.mean(rewards)

            policy_loss = []
            for lp, r in trajectories:
                # 强化学习核心：奖励高于平均则增加该概率，低于则减少
                policy_loss.append(-lp * (r - mean_reward if epi > 1 else r))

            self.agent_optimizer.zero_grad()
            final_loss = torch.stack(policy_loss).mean()
            final_loss.backward()
            self.agent_optimizer.step()

            # Detach hidden state to prevent backpropagating into the graph with modified parameters
            h_t = h_t.detach()

            print(f"Step [{t + 1}/{T}] | Avg Reward: {mean_reward:.4f} | Selected Avg: {selected_mask.sum().item()}")

            # Algorithm 1, Line 16: 随机打乱训练集样本顺序
            perm = torch.randperm(train_x.size(0))
            train_x, train_y = train_x[perm], train_y[perm]


# ==========================================
# 4. 模拟真实的不均衡故障数据
# ==========================================
def generate_imbalanced_data():
    # 模拟 43 维特征，4 类故障
    X, y = make_classification(n_samples=300, n_features=43, n_informative=30,
                               n_classes=4, weights=[0.7, 0.1, 0.1, 0.1],  # 极度不均衡
                               n_clusters_per_class=1, random_state=42)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    return X[:200], y[:200], X[200:], y[200:]  # 划分训练/验证


if __name__ == "__main__":
    trn_x, trn_y, val_x, val_y = generate_imbalanced_data()

    framework = DiagSelectFramework(feat_dim=43, num_classes=4)

    print(">>> 启动 DiagSelect 完整复现训练 (预计耗时较长，因为包含多次基础模型重训练)...")
    framework.run_offline_training(trn_x, trn_y, val_x, val_y, T=200, epi=3)