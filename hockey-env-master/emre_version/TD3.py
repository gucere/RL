import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim_1, hidden_dim_2, leaky_relu_grad=0.01):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.leaky_relu_grad = leaky_relu_grad
        self.fc1 = nn.Linear(state_dim, hidden_dim_1)
        self.ln1 = nn.LayerNorm(hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.ln2 = nn.LayerNorm(hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, action_dim)

        self.activation = nn.LeakyReLU(leaky_relu_grad)

        self.apply(self.init_weights)

    def forward(self, state):
        x = self.activation(self.ln1(self.fc1(state)))
        x = self.activation(self.ln2(self.fc2(x)))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=self.leaky_relu_grad, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2, leaky_relu_grad=0.01):
        super(Critic, self).__init__()

        self.leaky_relu_grad = leaky_relu_grad
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.ln1 = nn.LayerNorm(hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.ln2 = nn.LayerNorm(hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1)

        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.ln4 = nn.LayerNorm(hidden_dim_1)
        self.fc5 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.ln5 = nn.LayerNorm(hidden_dim_2)
        self.fc6 = nn.Linear(hidden_dim_2, 1)

        self.apply(self.init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        q1 = torch.nn.functional.leaky_relu(self.ln1(self.fc1(x)))
        q1 = torch.nn.functional.leaky_relu(self.ln2(self.fc2(q1)))
        q1 = self.fc3(q1)

        q2 = torch.nn.functional.leaky_relu(self.ln4(self.fc4(x)))
        q2 = torch.nn.functional.leaky_relu(self.ln5(self.fc5(q2)))
        q2 = self.fc6(q2)

        return q1, q2

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=self.leaky_relu_grad, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)


class TD3:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.05, policy_noise=0.2,
                  noise_clip=0.5, policy_freq=2, hidden_dim_1=256, hidden_dim_2=256,
                  actor_learning_rate=1e-3, critic_learning_rate=1e-3, weight_decay=1e-4,
                  grad_clip=1.0, leaky_relu_grad=0.01):

        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim_1, hidden_dim_2, leaky_relu_grad).to("cuda")
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim_1, hidden_dim_2, leaky_relu_grad).to("cuda")
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dim_1, hidden_dim_2, leaky_relu_grad).to("cuda")
        self.critic_target = Critic(state_dim, action_dim, hidden_dim_1, hidden_dim_2, leaky_relu_grad).to("cuda")
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.grad_clip = grad_clip  # Store gradient clipping value

        self.total_steps = 0


    def select_action(self, state, noise_std=0.1):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)
            action = self.actor(state_tensor)
            
            if noise_std > 0:
                noise = torch.randn_like(action, device="cuda") * noise_std
                action.add_(noise).clamp_(-1, 1)
            
            return action.cpu().numpy().flatten()

    def train(self, states, actions, next_states, rewards, dones, batch_size):
        states = torch.as_tensor(states, device="cuda", dtype=torch.float32)
        actions = torch.as_tensor(actions, device="cuda", dtype=torch.float32)
        next_states = torch.as_tensor(next_states, device="cuda", dtype=torch.float32)
        rewards = torch.as_tensor(rewards, device="cuda", dtype=torch.float32).unsqueeze(1)
        dones = torch.as_tensor(dones, device="cuda", dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            noise = torch.randn_like(actions, device="cuda") * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)

            q1_target, q2_target = self.critic_target(next_states, next_actions)
            target_q = rewards.view(-1, 1) + self.discount * (1 - dones.view(-1, 1)) * torch.min(q1_target, q2_target)

        q1, q2 = self.critic(states, actions)

        critic_loss = nn.MSELoss()(q1.squeeze(-1), target_q.squeeze(-1)) + nn.MSELoss()(q2.squeeze(-1), target_q.squeeze(-1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        self.total_steps += 1

        if self.total_steps % self.policy_freq == 0:
            actor_loss = -self.critic(states, self.actor(states))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

        self.update_target_networks()

    def update_target_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
