import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy  # 用于深拷贝最佳模型权重
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from utils.dataLoad import load_YiChang_with_classes
from models.RacAdvancedClassifier import RacAdvancedClassifier


# ==========================================
# 1. 智能体模型 (Section III-B, Eq. 6-11)
# ==========================================
class DiagSelectAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):  # 论文 Section IV-A-1: H=80
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
# 3. 核心算法类 (Algorithm 1 & Section III-D)
# ==========================================
class DiagSelectFramework:
    # agent的学习率设置为0.001
    def __init__(self, feat_dim, num_classes, agent_lr=1e-3, base_lr=1e-2):
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.agent = DiagSelectAgent(input_dim=feat_dim + num_classes)

        # 论文 Section IV-A-1: Agent 使用 RMSprop 优化器
        self.agent_optimizer = optim.RMSprop(self.agent.parameters(), lr=agent_lr)

        # --- [新增监控变量] ---
        self.reward_history = []
        self.best_reward = -float('inf')
        self.best_agent_state = None

    def _calculate_g_mean(self, y_true, y_pred):
        """
        对应 Section III-D: 评价指标 G-Mean
        """
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        with np.errstate(divide='ignore', invalid='ignore'):
            recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-7)  # 加上 epsilon 防止除零
            recalls = np.nan_to_num(recalls)
        return np.exp(np.mean(np.log(recalls + 1e-7)))

    def pre_train_agent(self, train_x, train_y, epochs=10):
        """
        对应论文 Section III-B, Eq.(1): 预训练 Agent 使其初始概率对齐类别分布
        """
        print(">>> 正在进行 Agent 预训练...")
        self.agent.train()

        # 计算每个类别的样本比例，用于构造目标概率 p_ini
        counts = torch.bincount(train_y)
        probs_target = counts.float() / len(train_y)

        # 构造目标分布：每个样本被选中的初始概率 = 其所属类别的占比
        # target_p_ini shape: [num_samples, 2] -> [不选概率, 选中概率]
        target_p_ini = torch.zeros(len(train_y), 2)
        for i in range(len(train_y)):
            p_class = probs_target[train_y[i]]
            target_p_ini[i, 1] = p_class  # 第二维是选择概率
            target_p_ini[i, 0] = 1 - p_class  # 第一维是不选概率

        optimizer = optim.Adam(self.agent.parameters(), lr=1e-3)
        h_0 = torch.zeros(1, 1, self.agent.hidden_dim)

        for epoch in range(epochs):
            y_onehot = F.one_hot(train_y, num_classes=self.num_classes).float()
            s_t = torch.cat([train_x, y_onehot], dim=1).unsqueeze(0)

            # 获取 Agent 当前输出
            probs, _ = self.agent(s_t, h_0)

            # 计算公式 (1) 的交叉熵损失
            loss = -torch.mean(torch.sum(target_p_ini * torch.log(probs + 1e-7), dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"  Pre-train Epoch [{epoch + 1}/{epochs}] | Loss: {loss.item():.4f}")

    def train_base_model(self, train_x, train_y, val_x, val_y):
        """
        使用 RacAdvancedClassifier 进行训练和评估
        """
        # 转换为 numpy 格式
        # RacAdvancedClassifier 需要 1-based labels (1, 2, 3...)
        # train_y 和 val_y 假设是 0-based
        X_train = train_x.cpu().numpy()
        y_train = train_y.cpu().numpy() + 1
        X_val = val_x.cpu().numpy()
        y_true = val_y.cpu().numpy()

        # 检查类别数量，防止无法训练
        # 如果选出的样本类别太少，可能导致 RAC 内部逻辑(如 OPW 需要 1,2 类) 报错
        # 但 RAC 内部也有处理。这里做个简单保护
        if len(np.unique(y_train)) < 2:
            return 0.0

        try:
            # 初始化并训练分类器
            # verbose=False 以减少输出
            clf = RacAdvancedClassifier(kernel_type='rbf', kernel_pars=[10.7], c=7.5, verbose=False)
            clf.fit(X_train, y_train)

            # 预测
            preds = clf.predict(X_val)

            # 还原为 0-based 用于计算指标
            preds_0based = preds - 1

            # 论文关键：使用 Macro-F1 作为主要奖励信号 (Eq. 14)
            f1 = f1_score(y_true, preds_0based, average='macro')
            gmean = self._calculate_g_mean(y_true, preds_0based)
            # 综合奖励 (可根据论文具体偏好调整比例)
            reward = 0.5 * f1 + 0.5 * gmean
            return reward
        except Exception as e:
            # print(f"Training failed: {e}")
            return 0.0

    def run_offline_training(self, train_x, train_y, val_x, val_y, T=20, epi=5, patience=8):
        """
        严格执行 Algorithm 1: Offline Training
        patience: [新增] 忍受奖励不增长的最大步数
        """
        # Algorithm 1, Line 3: 初始化隐状态
        h_t = torch.zeros(1, 1, self.agent.hidden_dim)

        no_improve_counter = 0  # [新增] 早停计数器

        for t in range(T):
            # Algorithm 1, Line 6: 构造状态 s_t (特征+标签)
            y_onehot = F.one_hot(train_y, num_classes=self.num_classes).float()
            s_t = torch.cat([train_x, y_onehot], dim=1).unsqueeze(0)

            # Algorithm 1, Line 7: 获取选择概率
            # 此时计算图开始记录
            probs, h_t = self.agent(s_t, h_t)

            trajectories = []  # 记录 epi 次采样的结果
            total_selected_count = 0

            for k in range(epi):
                # Eq. (12): 采样动作
                m = torch.distributions.Categorical(probs)
                actions = m.sample()
                log_probs = m.log_prob(actions).sum()

                # Algorithm 1, Line 10: 筛选子集
                selected_mask = (actions == 1)
                selected_count = selected_mask.sum().item()
                total_selected_count += selected_count

                if selected_count < self.num_classes * 2:  # 保护机制：样本太少无法训练
                    reward = 0.0
                else:
                    sel_x = train_x[selected_mask]
                    sel_y = train_y[selected_mask]

                    # 直接传递数据给 train_base_model
                    reward = self.train_base_model(sel_x, sel_y, val_x, val_y)

                trajectories.append((log_probs, reward))

            avg_selected = total_selected_count / epi

            # --- 策略梯度更新 (Eq. 15) ---
            # 计算 epi 次采样的平均奖励作为 Baseline 以减小方差
            rewards = [tr[1] for tr in trajectories]
            mean_reward = np.mean(rewards)
            self.reward_history.append(mean_reward)  # [新增记录]

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

            # --- [新增早停与监控逻辑] ---
            status_msg = ""
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.best_agent_state = copy.deepcopy(self.agent.state_dict())
                no_improve_counter = 0
                status_msg = " (Best ⭐)"
            else:
                no_improve_counter += 1
                status_msg = f" (No improve {no_improve_counter}/{patience})"

            print(f"Step [{t + 1}/{T}] | Avg Reward: {mean_reward:.4f} | Selected Avg: {avg_selected:.1f}{status_msg}")

            if no_improve_counter >= patience:
                print(f">>> 触发早停机制，加载第 {t + 1 - patience} 步的最佳参数。")
                break

            # Algorithm 1, Line 16: 随机打乱训练集样本顺序
            perm = torch.randperm(train_x.size(0))
            train_x, train_y = train_x[perm], train_y[perm]

        # 训练结束，恢复最优 Agent
        if self.best_agent_state is not None:
            self.agent.load_state_dict(self.best_agent_state)


# ==========================================
# 4. 数据加载与处理
# ==========================================
def load_and_process_data():
    df = load_YiChang_with_classes()

    # 假设 ZH_CLASS 是标签，且为 1, 2, 3
    # 转换为 0-based: 0, 1, 2
    # 注意：RacAdvancedClassifier 内部我们做了一些适配，这里只要保证 train_y 是 0,1,2 即可
    y = df['ZH_CLASS'].values - 1
    X = df.drop(columns=['ZH_CLASS']).values

    # 划分数据集: 训练集 0.7, 验证集 0.1, 测试集 0.2
    # 第一次划分：分出测试集 (0.2)
    # stratify确保类别分布一致
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 第二次划分：从剩余的 (0.8) 中分出验证集 (0.1 / 0.8 = 0.125)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
    )

    # 转换为 Tensor
    trn_x = torch.FloatTensor(X_train)
    trn_y = torch.LongTensor(y_train)
    val_x = torch.FloatTensor(X_val)
    val_y = torch.LongTensor(y_val)
    tst_x = torch.FloatTensor(X_test)
    tst_y = torch.LongTensor(y_test)

    return trn_x, trn_y, val_x, val_y, tst_x, tst_y


if __name__ == "__main__":
    # 1. 准备数据
    trn_x, trn_y, val_x, val_y, tst_x, tst_y = load_and_process_data()

    feat_dim = trn_x.shape[1]
    num_classes = len(torch.unique(trn_y))

    print(f"Data Loaded: Train {trn_x.shape}, Val {val_x.shape}, Test {tst_x.shape}")
    print(f"Features: {feat_dim}, Classes: {num_classes}")

    # 2. 训练 Agent (老师学习怎么挑数据)
    framework = DiagSelectFramework(feat_dim=feat_dim, num_classes=num_classes)

    # --- 新增预训练调用 ---
    framework.pre_train_agent(trn_x, trn_y, epochs=20)

    print("\n>>> Phase 1: 训练 Agent (寻找最佳筛选策略)...")
    # 建议将 T 设大一点（比如 50），由 patience 自动控制结束
    framework.run_offline_training(trn_x, trn_y, val_x, val_y, T=20, epi=3, patience=800)

    # =======================================================
    # Phase 2: 使用训练好的 Agent 全局筛选数据 (Train + Val)
    # =======================================================
    print("\n>>> Phase 2: 使用训练好的最佳 Agent 全局筛选数据 (Train + Val)...")

    # 将 Agent 设置为评估模式
    framework.agent.eval()


    def select_best_subset(agent, x, y, num_classes):
        """辅助函数：用 Agent 筛选数据"""
        with torch.no_grad():
            # 构造状态输入
            y_onehot = F.one_hot(y, num_classes=num_classes).float()
            s_t = torch.cat([x, y_onehot], dim=1).unsqueeze(0)
            h_init = torch.zeros(1, 1, agent.hidden_dim)

            # 获取概率
            probs, _ = agent(s_t, h_init)

            # 贪婪选择: 概率 > 0.5 的样本
            selected_mask = (probs[:, 1] > 0.5)

            # 如果筛选太少，就全部保留 (兜底策略)
            if selected_mask.sum() < num_classes * 2:
                print("  Warning: Agent 选得太少，保留全集")
                return x, y

            return x[selected_mask], y[selected_mask]


    # 1. 从原训练集中挑最好的
    final_trn_x, final_trn_y = select_best_subset(framework.agent, trn_x, trn_y, num_classes)
    # 2. 从原验证集中挑最好的
    final_val_x, final_val_y = select_best_subset(framework.agent, val_x, val_y, num_classes)

    # 3. 合并
    combined_X = torch.cat([final_trn_x, final_val_x], dim=0)
    combined_y = torch.cat([final_trn_y, final_val_y], dim=0)

    print(f"  原始总数: {len(trn_x) + len(val_x)}")
    print(f"  筛选后总数: {len(combined_X)} (Train选出 {len(final_trn_x)} + Val选出 {len(final_val_x)})")

    # =======================================================
    # Phase 3: 在测试集上进行最终测试
    # =======================================================
    print("\n>>> Phase 3: 训练最终分类器并测试...")

    # 转换 tensor -> numpy 用于 RacAdavancedClassifier
    X_train_final = combined_X.numpy()
    y_train_final = combined_y.numpy() + 1  # 0-based -> 1-based

    X_test_final = tst_x.numpy()
    y_test_final = tst_y.numpy()

    # 最终训练
    clf_final = RacAdvancedClassifier(kernel_type='rbf', kernel_pars=[10.7], c=7.5, verbose=True)
    clf_final.fit(X_train_final, y_train_final)

    # 最终预测
    preds = clf_final.predict(X_test_final)
    preds_0based = preds - 1

    # 计算最终指标
    final_f1 = f1_score(y_test_final, preds_0based, average='macro')
    # confusion_matrix 需要 ensure labels
    cm = confusion_matrix(y_test_final, preds_0based, labels=range(num_classes))
    print(cm)
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-7)
        recalls = np.nan_to_num(recalls)
    final_gmean = np.exp(np.mean(np.log(recalls + 1e-7)))

    print("=" * 40)
    print(f"Final Test Result (After DiagSelect):")
    print(f"Macro F1 Score : {final_f1:.4f}")
    print(f"G-Mean Score   : {final_gmean:.4f}")
    print("=" * 40)