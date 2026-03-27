import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from copy import deepcopy

class DiagSelectAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(DiagSelectAgent, self).__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h):
        h_next = self.gru(x, h)
        logits = self.fc(h_next)
        probs = torch.softmax(-logits, dim=-1) # As per eq 11 in paper: exp(-d) / sum exp(-d)
        return probs, h_next

class DiagSelectResampler:
    def __init__(self, 
                 hidden_dim=5, 
                 pre_lr=0.01, 
                 rl_lr=0.001, 
                 pre_epochs=50, 
                 rl_episodes=200, 
                 rl_steps_per_episode=10, 
                 early_stop=20,
                 device='cpu'):
        """
        Hyperparameters for DiagSelect:
        - hidden_dim: hidden state size of GRU (e.g. 5 for small/synthetic, 80 for complex)
        - rl_steps_per_episode: 'epi' in paper, times to sample actions per time step to estimate expectations
        """
        self.hidden_dim = hidden_dim
        self.pre_lr = pre_lr
        self.rl_lr = rl_lr
        self.pre_epochs = pre_epochs
        self.rl_episodes = rl_episodes
        self.rl_steps_per_episode = rl_steps_per_episode
        self.early_stop = early_stop
        self.device = device
        self.agent = None

    def fit(self, X_trn, y_trn, X_val, y_val, classifier_cls, **classifier_kwargs):
        """
        Train the DiagSelect agent to find the optimal sampling policy.
        """
        N, num_features = X_trn.shape
        num_classes = len(np.unique(y_trn))
        
        class_counts = {c: np.sum(y_trn == c) for c in np.unique(y_trn)}
        
        p_ini = np.zeros((N, 2))
        for c, count in class_counts.items():
            idx = (y_trn == c)
            # Use inverse proportion to encourage selecting minority and dropping majority
            prop = count / N
            p_ini[idx, 1] = 1.0 - prop
            p_ini[idx, 0] = prop
            
        p_ini_tensor = torch.tensor(p_ini, dtype=torch.float32).to(self.device)
        
        X_t = torch.tensor(X_trn, dtype=torch.float32)
        y_t_onehot = torch.zeros(N, num_classes)
        # Assuming y_trn is 0-indexed contiguous integer labels
        y_tensor = torch.tensor(y_trn, dtype=torch.int64)
        y_t_onehot.scatter_(1, y_tensor.unsqueeze(1), 1)
        
        s_t = torch.cat([X_t, y_t_onehot], dim=1).to(self.device)
        
        input_dim = num_features + num_classes
        self.agent = DiagSelectAgent(input_dim, self.hidden_dim).to(self.device)
        
        # 1. Pretraining (Cross Entropy to guide initial selecting probability)
        optimizer_pre = optim.Adam(self.agent.parameters(), lr=self.pre_lr)
        
        h = torch.zeros(N, self.hidden_dim).to(self.device)
        for ep in range(self.pre_epochs):
            optimizer_pre.zero_grad()
            probs, h = self.agent(s_t, h.detach())
            loss = -torch.mean(torch.sum(p_ini_tensor * torch.log(probs + 1e-8), dim=1))
            loss.backward()
            optimizer_pre.step()
            
        # 2. RL Training (REINFORCE)
        optimizer_rl = optim.RMSprop(self.agent.parameters(), lr=self.rl_lr)
        best_reward = -1.0
        best_agent_state = None
        no_improve = 0
        
        h = torch.zeros(N, self.hidden_dim).to(self.device)
        
        for ep in range(self.rl_episodes):
            optimizer_rl.zero_grad()
            
            probs, h = self.agent(s_t, h.detach())
            
            total_loss = 0
            ep_reward = 0
            
            # Use baseline to stabilize REINFORCE
            collected_rewards = []
            collected_log_probs = []
            
            dist = torch.distributions.Categorical(probs)
            
            for step in range(self.rl_steps_per_episode):
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                action_np = action.cpu().numpy()
                selected_idx = np.where(action_np == 1)[0]
                
                reward = 0.0
                if len(selected_idx) > 0 and len(np.unique(y_trn[selected_idx])) == num_classes:
                    X_sub = X_trn[selected_idx]
                    y_sub = y_trn[selected_idx]
                    
                    clf = classifier_cls(**classifier_kwargs)
                    clf.fit(X_sub, y_sub)
                    
                    y_val_pred = clf.predict(X_val)
                    reward = f1_score(y_val, y_val_pred, average='macro')
                
                ep_reward += reward
                collected_rewards.append(reward)
                collected_log_probs.append(torch.sum(log_prob))
                
            avg_reward = ep_reward / self.rl_steps_per_episode
            baseline = avg_reward # simple baseline
            
            for r, lp in zip(collected_rewards, collected_log_probs):
                adv = r - (baseline * 0.9) # leave some advantage strictly positive for good samples
                total_loss -= adv * lp
                
            total_loss /= self.rl_steps_per_episode
            total_loss.backward()
            optimizer_rl.step()
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_agent_state = deepcopy(self.agent.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                
            if (ep + 1) % 10 == 0:
                print(f"RL Epoch {ep+1}/{self.rl_episodes}, Reward: {avg_reward:.4f}")
                
            if no_improve >= self.early_stop:
                print(f"Early stopped at epoch {ep+1}")
                break
                
        if best_agent_state is not None:
            self.agent.load_state_dict(best_agent_state)
            
        return self

    def resample(self, X_trn, y_trn):
        if self.agent is None:
            raise ValueError("Agent is not trained. Call fit() first.")
            
        N, num_features = X_trn.shape
        num_classes = len(np.unique(y_trn))
        
        X_t = torch.tensor(X_trn, dtype=torch.float32)
        y_t_onehot = torch.zeros(N, num_classes)
        y_tensor = torch.tensor(y_trn, dtype=torch.int64)
        y_t_onehot.scatter_(1, y_tensor.unsqueeze(1), 1)
        
        s_t = torch.cat([X_t, y_t_onehot], dim=1).to(self.device)
        
        self.agent.eval()
        with torch.no_grad():
            h = torch.zeros(N, self.hidden_dim).to(self.device)
            probs, h = self.agent(s_t, h)
            # Use sampling instead of argmax because RL optimized the sampling distribution
            # to yield a balanced subset. If majority prob is 0.1, argmax will drop all of them.
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().cpu().numpy()
            
        selected_idx = np.where(action == 1)[0]
        if len(selected_idx) == 0:
            print("Warning: Agent selected 0 samples. Returning original set.")
            return X_trn, y_trn
            
        return X_trn[selected_idx], y_trn[selected_idx]
