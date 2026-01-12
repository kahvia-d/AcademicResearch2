import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from utils.dataLoad import load_YiChang_with_classes

# 设置随机种子以保证结果可复现
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# ==========================================
# 1. 论文核心组件定义: DiagSelect Agent
# 参考论文 Section III-B Model Framework
# ==========================================
class DiagSelectAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        """
        初始化 DiagSelect 智能体
        input_dim: 特征维度 + 标签维度 (论文公式 5: X + onehot(Y))
        """
        super(DiagSelectAgent, self).__init__()
        # 论文公式 6-9: 使用 GRU 进行状态编码
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 论文公式 10-11: 全连接层输出动作概率 (选 或 不选)
        self.fc = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, state):
        """
        前向传播
        state: Tensor shape (Batch_Size, Sequence_Length, Input_Dim)
        这里的 Sequence_Length 即为训练集样本数量
        """
        # GRU 输出
        # h_out shape: (Batch, Seq_Len, Hidden_Dim)
        h_out, _ = self.gru(state)

        # 计算动作概率
        logits = self.fc(h_out)
        probs = self.softmax(logits)  # shape: (Batch, Seq_Len, 2)

        return probs


# ==========================================
# 2. 辅助函数与环境模拟
# ==========================================
def get_reward(selected_indices, X_train, y_train, X_val, y_val):
    """
    计算奖励 (Reward) - 论文公式 13-14
    环境：诊断模型 (这里使用 SVM，如论文案例所示)
    奖励：验证集上的 F1-score
    """
    if len(selected_indices) < 2 or len(np.unique(y_train[selected_indices])) < 2:
        return 0.0  # 如果样本太少或只选了一类，奖励为0

    # 获取被选中的子集
    X_subset = X_train[selected_indices]
    y_subset = y_train[selected_indices]

    # 训练诊断模型 (Diagnosis Model)
    # 论文中提到 DiagSelect 与模型解耦，这里使用 SVM
    clf = SVC(kernel='linear', random_state=seed)
    clf.fit(X_subset, y_subset)

    # 在验证集上预测
    y_pred = clf.predict(X_val)

    # 计算奖励 (Macro F1)
    reward = f1_score(y_val, y_pred, average='macro')
    return reward


def prepare_state(X, y, num_classes):
    """
    准备状态 s_t = X_trn ⊕ onehot(Y_trn) - 论文公式 5
    """
    # One-hot 编码标签
    y_onehot = np.eye(num_classes)[y]
    # 拼接特征和标签
    state_np = np.hstack((X, y_onehot))
    # 转换为 Tensor，增加 Batch 维度 (Batch=1)
    state_tensor = torch.FloatTensor(state_np).unsqueeze(0)
    return state_tensor


# ==========================================
# 3. 主流程：数据生成、训练与测试
# ==========================================
def main():
    print("正在加载真实不平衡数据...")
    # 1. 加载真实不平衡数据集
    df = load_YiChang_with_classes()
    y = df['ZH_CLASS'].values - 1
    X = df.drop('ZH_CLASS', axis=1).values
    num_classes = len(np.unique(y))

    # 数据集划分：训练集、验证集、测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=seed,
                                                      stratify=y_train_full)

    # 标准化 (论文 Step 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"训练集大小: {len(X_train)} (类别分布: {np.bincount(y_train)})")
    print(f"验证集大小: {len(X_val)}")

    # 参数设置
    input_dim = X_train.shape[1] + num_classes  # 特征数 + 类别数
    hidden_dim = 64
    learning_rate = 0.001
    episodes = 50  # 对应论文中的 Time steps / Episodes

    # 初始化 Agent 和 优化器
    agent = DiagSelectAgent(input_dim, hidden_dim)
    optimizer = optim.RMSprop(agent.parameters(), lr=learning_rate)  # 论文 Case Study 使用 RMSprop

    # ------------------------------------------
    # Step 2: 预训练 (Pretrain) - 可选
    # 论文公式 1: 使用交叉熵损失，基于初始概率预训练
    # 这里为了简化，直接进入强化学习训练阶段，但在逻辑中包含了预训练的概率初始化思想
    # ------------------------------------------

    print("\n开始 DiagSelect 强化学习训练...")

    # 记录最佳模型
    best_reward = 0
    best_subset_indices = None

    for episode in range(episodes):
        # 论文提到每个 time step 可以 shuffle 样本，这里简化为固定顺序或手动 shuffle
        # 构建状态
        state = prepare_state(X_train, y_train, num_classes=num_classes)

        # 获取动作概率 (Batch=1, Seq_Len=N, Actions=2)
        # probs[:, :, 0]是不选的概率, probs[:, :, 1]是选的概率
        probs = agent(state)

        # 动作采样 (Sample Action) - 论文公式 12
        # 构建分布进行采样
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()  # shape: (1, N)

        # 获取 log probability 用于计算损失
        log_probs = dist.log_prob(actions)

        # 解析动作：1 代表选择，0 代表不选
        selected_mask = actions.squeeze(0).cpu().numpy()
        selected_indices = np.where(selected_mask == 1)[0]

        # 计算奖励 (Reward)
        reward = get_reward(selected_indices, X_train, y_train, X_val, y_val)

        # 更新最佳策略记录
        if reward > best_reward:
            best_reward = reward
            best_subset_indices = selected_indices

        # 计算 Loss - REINFORCE 算法 (论文公式 15)
        # Loss = - sum(reward * log_prob)
        # 注意：这里我们最大化奖励，所以 Loss 取负
        loss = -torch.sum(log_probs * reward)

        # 反向传播与优化 (论文公式 16)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{episodes}, Selected: {len(selected_indices)}, Reward (Val F1): {reward:.4f}")
    # ------------------------------------------
    # Step 4 & 5: 在线测试 (Online Testing)
    # ------------------------------------------
    print("\n训练完成。进行最终测试...")

    # 使用最佳策略选出的子集训练最终的诊断模型
    if best_subset_indices is None or len(best_subset_indices) == 0:
        print("警告：未选择有效样本，使用全量数据兜底。")
        best_subset_indices = range(len(X_train))

    X_final_train = X_train[best_subset_indices]
    y_final_train = y_train[best_subset_indices]

    # 最终诊断模型 (Online Fault Diagnosis Model)
    final_clf = SVC(kernel='linear', random_state=seed)
    final_clf.fit(X_final_train, y_final_train)

    # 在测试集上评估
    y_test_pred = final_clf.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    # 对比基准 (使用所有原始数据训练)
    baseline_clf = SVC(kernel='linear', random_state=seed)
    baseline_clf.fit(X_train, y_train)
    y_base_pred = baseline_clf.predict(X_test)
    base_f1 = f1_score(y_test, y_base_pred, average='macro')

    print("-" * 30)
    print(f"DiagSelect 选出的样本数: {len(X_final_train)} / {len(X_train)}")
    print(f"原始不平衡数据训练 F1: {base_f1:.4f}")
    print(f"DiagSelect 优化后训练 F1: {test_f1:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
