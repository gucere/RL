import numpy as np
import torch
import os
import json
from emre_version.TD3 import TD3
from emre_version.memory import Memory
import hockey.hockey_env as h_env
from datetime import datetime

class Test:
    def __init__(self, opponents, discount, tau, policy_noise, noise_clip, policy_freq,
                 hidden_dim_1=2048, hidden_dim_2=2048, learning_rate=1e-3, weight_decay=1e-4, grad_clip=1.0,
                 total_episodes=100, number_of_games=10, batch_size=256, episode_length=200, exploration_noise=0.1,
                 min_exploration_noise=0.0001, exploration_decay=0.999999, whether_to_render=False, memory_limit=1000000, train_against_weak_n_number_of_times=1000):

        # Training only in NORMAL mode since testing for full game seemed more optimal
        self.env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        #Create agent
        self.agent = TD3(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            discount=discount,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip
        )

        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Load existing memory or create new one
        self.memory_limit = memory_limit
        self.memory = self.load_memory()
        self.opponents = opponents
        self.obs, _info = self.env.reset()

        # Training parameters
        self.total_episodes = total_episodes
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.exploration_noise = exploration_noise
        self.min_exploration_noise = min_exploration_noise
        self.exploration_decay = exploration_decay
        self.train_against_weak_n_number_of_times = train_against_weak_n_number_of_times

        # Used for both training and testing
        self.number_of_games = number_of_games
        self.whether_to_render = whether_to_render
    
        # Load or initialize training history
        self.training_history = self.load_training_history()
        
        # Load the all-time best reward
        self.best_reward_ever = self.training_history.get("best_reward_ever", -float('inf'))

    def load_training_history(self):
        """Load training history from file"""
        history_path = "models/training_history.json"
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    return history
            except Exception as e:
                print(f"Error loading training history: {e}")
        
        # Initialize new history
        return {
            "episodes": [],
            "best_reward_ever": -float('inf'),
            "total_training_episodes": 0
        }

    def save_training_history(self):
        """Save training history to file"""
        with open("models/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)

    def save_memory(self):
        """Save memory buffer to file"""
        try:
            transitions = self.memory.get_all_transitions()
            if len(transitions) > 0:
                torch.save(transitions, "models/memory_buffer.pt")
        except Exception as e:
            print(f"Error saving memory: {e}")

    def load_memory(self):
        memory_path = "models/memory_buffer.pt"
        if os.path.exists(memory_path):
            try:
                transitions = torch.load(memory_path, map_location="cuda", weights_only=False)
                # Enforce new limit - it may be needed if user(s) decide to increase or decrease memory limit
                memory = Memory(max_size=self.memory_limit)
                
                if len(transitions) > memory.max_size:
                    transitions = transitions[-memory.max_size:]

                for transition in transitions:
                    memory.add_transition(transition)

                print(f"Loaded memory buffer with {memory.size} transitions (Limited to {memory.max_size})")
                return memory
            except Exception as e:
                print(f"Error loading memory buffer: {e}")
        return Memory(max_size=self.memory_limit)


    def render_episode(self, player2, play_best_agent=False, render_current=True):
        with torch.no_grad():
            torch.cuda.empty_cache()
            self.obs, _info = self.env.reset()
            episode_reward = 0
            while True:
                if render_current:
                    _ = self.env.render()
                action1, action2, combined_action = self.get_action(player2)
                self.obs, reward, done, trunc, _info = self.env.step(combined_action)
                episode_reward += reward
                if done or trunc:
                    break
            # commented out this since it not must have statement. Keeping this for potential future users
            # if play_best_agent:
            #     print(f"Episode Reward with Best Agent: {episode_reward}")
            return episode_reward
        
    def save(self, actor_path, critic_path):
        actor_state = {k: v.cpu() for k, v in self.agent.actor.state_dict().items()}
        critic_state = {k: v.cpu() for k, v in self.agent.critic.state_dict().items()}
        torch.save(actor_state, actor_path)
        torch.save(critic_state, critic_path)

    def load_agent(self, actor_path, critic_path):
        torch.cuda.empty_cache()
        if os.path.exists(actor_path):
            self.agent.actor.load_state_dict(torch.load(actor_path, weights_only=True))  
            self.agent.critic.load_state_dict(torch.load(critic_path, weights_only=True))
            self.agent.actor.to("cuda")
            self.agent.critic.to("cuda")
            print(f"Loaded agent from {actor_path} and {critic_path}")
            return True
        else:
            print(f"Could not find model at {actor_path}")
            return False


    def get_action(self, player2):
        action1 = torch.tensor(self.agent.select_action(self.obs), device='cuda')
        noise = torch.normal(0, self.exploration_noise, size=(4,), device='cuda')
        action1 = action1[:4] + noise # note to reader/grader: select_action returns an action for both players
        low = torch.tensor(self.env.action_space.low[:4], device='cuda')
        high = torch.tensor(self.env.action_space.high[:4], device='cuda')
        action1 = torch.clamp(action1, min=low, max=high)
        action2 = torch.tensor(player2.act(self.env.obs_agent_two()), device='cuda')
        combined_action = torch.cat([action1, action2]).cpu().numpy()
        return action1.cpu().numpy(), action2.cpu().numpy(), combined_action

    def evaluate_agent(self, number_of_games=10, whether_to_render=False):
        torch.cuda.empty_cache()
        results = []
        with torch.no_grad():
            with torch.cuda.device("cuda"):
                for idx, player2 in enumerate(self.opponents):
                    opponent_type = "weak" if idx == 0 else "strong"
                    total_reward = 0
                    successes = 0
                    
                    #print(f"\nTesting against {opponent_type} opponent:") # optional print statement
                    for eval_ep in range(number_of_games):
                        reward = self.render_episode(player2, play_best_agent=True, render_current=whether_to_render)
                        total_reward += reward
                        if reward > 0:
                            successes += 1
                    
                    avg_reward = total_reward / number_of_games
                    success_rate = (successes / number_of_games) * 100
                    
                    results.append({
                        'opponent_type': opponent_type,
                        'avg_reward': avg_reward,
                        'success_rate': success_rate
                    })
        overall_avg_reward = sum(r['avg_reward'] for r in results) / len(results)
        return overall_avg_reward
    
    def train(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory at start
        session_best_reward = -float('inf')
        
        # Load the best agent if it exists
        if os.path.exists("models/best_actor.pth"):
            self.load_agent("models/best_actor.pth", "models/best_critic.pth")

            # Evaluate the best model inputed number_of_games times
            print("\nEvaluating best saved agent before training...")
            session_best_reward = self.evaluate_agent(number_of_games=self.number_of_games, whether_to_render=False)
            
            # Update session best reward if we loaded a better model
            # Use max for more stability, use the current version for more testing
            self.best_reward_ever = session_best_reward #max(self.best_reward_ever, session_best_reward)
            print(f"All-time best reward: {self.best_reward_ever}")
        # Get current training progress
        global_episode = self.training_history.get("total_training_episodes", 0)

        # Main training loop
        print("\nStarting mixed opponent training...")
        for episode in range(self.total_episodes):
            # Alternate between opponents for better training
            if global_episode < self.train_against_weak_n_number_of_times:  # First train_against_weak_n_number_of_times episodes
                player2 = self.opponents[0]  # Use only weak opponent
                opponent_type = "weak"
            else:
                player2_idx = global_episode % len(self.opponents)
                player2 = self.opponents[player2_idx]
                opponent_type = "weak" if player2_idx == 0 else "strong"
            
            self.obs, _info = self.env.reset()
            episode_reward = 0

            for step in range(self.episode_length):
                self.exploration_noise = max(self.min_exploration_noise, self.exploration_noise * self.exploration_decay) # exploration noise decay
                action1, action2, combined_action = self.get_action(player2)
                next_state, reward, done, trunc, _info = self.env.step(combined_action)
                self.memory.add_transition([self.obs, combined_action, next_state, reward, done])

                if self.memory.size > self.batch_size:
                    batch = self.memory.sample(self.batch_size)
                    device = torch.device("cuda")
                    
                    states, actions, next_states, rewards, dones = zip(*batch)
                    states = torch.tensor(np.vstack([state.cpu() if torch.is_tensor(state) else state for state in states]), dtype=torch.float32, device=device)
                    actions = torch.tensor(np.vstack([action.cpu() if torch.is_tensor(action) else action for action in actions]), dtype=torch.float32, device=device)
                    next_states = torch.tensor(np.vstack([next_state.cpu() if torch.is_tensor(next_state) else next_state for next_state in next_states]), dtype=torch.float32, device=device)
                    rewards = torch.tensor(np.array([reward.cpu() if torch.is_tensor(reward) else reward for reward in rewards], dtype=np.float32), dtype=torch.float32, device=device)
                    dones = torch.tensor(np.array([done.cpu() if torch.is_tensor(done) else done for done in dones], dtype=np.float32), dtype=torch.float32, device=device)
                    self.agent.train(states, actions, next_states, rewards, dones, self.batch_size)

                self.obs = next_state
                episode_reward += reward
                if done or trunc:
                    break

            # Update episode tracking
            global_episode += episode + 1
            
            # Record episode details
            episode_data = {
                "episode": global_episode,
                "opponent_type": opponent_type,
                "reward": episode_reward,
                "timestamp": datetime.now().isoformat()
            }
            self.training_history["episodes"].append(episode_data)
            self.training_history["total_training_episodes"] = global_episode
            
            
            should_evaluate = (
                episode % 100 == 0 or  # Regular checkpoint (just in case or in case if it is the first ever)
                episode_reward > self.best_reward_ever or # If the episode looks promising to evaluate
                True
            )
            # Evaluations take time and computational power, intend to evaluate only promising ones
            if should_evaluate:
                print("testing for episode with singular test reward: ", episode_reward)
                avg_reward = self.evaluate_agent(number_of_games=self.number_of_games, whether_to_render=False)
                print("new = ", avg_reward, " vs best so far = ", self.best_reward_ever)
                if avg_reward > self.best_reward_ever:
                    self.best_reward_ever = avg_reward
                    self.training_history["best_reward_ever"] = self.best_reward_ever
                    print(f"New all-time best reward: {self.best_reward_ever}. Saving model...")
                        
                    # Save best model
                    self.save("models/best_actor.pth", "models/best_critic.pth")
                    print("saved new model")
                        
                    # Save a record of this version
                    with open("models/metadata.json", "w") as f:
                        json.dump({
                            "version": "latest",
                            "reward": self.best_reward_ever,
                            "episode": episode,
                            "opponent_type": "evaluation",
                            "timestamp": datetime.now().isoformat()
                        }, f, indent=2)         
            # Periodically save training history and memory
            if episode % 5 == 0 or episode == self.total_episodes - 1:
                self.save_training_history()
                self.save_memory()

        # Save final model regardless of performance
        self.save("models/actor_final.pth", "models/critic_final.pth")
        
        # Final save of history and memory
        self.save_training_history()
        self.save_memory()
        
        print(f"Training complete. Final all-time best reward: {self.best_reward_ever}")

    def run_best_model(self, number_of_games=10, whether_to_render=False):
        self.load_agent("models/best_actor.pth", "models/best_critic.pth")
        self.evaluate_agent(number_of_games=number_of_games, whether_to_render=whether_to_render)

def main(user_variables):
    test_object = Test(
        opponents=user_variables.get("opponents"),
        discount=user_variables.get("discount"),
        tau=user_variables.get("tau"),
        policy_noise=user_variables.get("policy_noise"),
        noise_clip=user_variables.get("noise_clip"),
        policy_freq=user_variables.get("policy_freq"),
        hidden_dim_1=user_variables.get("hidden_dim_1"),
        hidden_dim_2=user_variables.get("hidden_dim_2"),
        learning_rate=user_variables.get("learning_rate"),
        weight_decay=user_variables.get("weight_decay"),
        grad_clip=user_variables.get("grad_clip"),
        total_episodes=user_variables.get("total_episodes"),
        number_of_games=user_variables.get("number_of_games"),
        batch_size=user_variables.get("batch_size"),
        episode_length=user_variables.get("episode_length"),
        exploration_noise=user_variables.get("exploration_noise"),
        min_exploration_noise=user_variables.get("min_exploration_noise"),
        exploration_decay=user_variables.get("exploration_decay"),
        whether_to_render=user_variables.get("whether_to_render"),
        memory_limit=user_variables.get("memory_limit"),
        train_against_weak_n_number_of_times=user_variables.get("train_against_weak_n_number_of_times")
    )

    if user_variables.get("whether_to_train"):
        test_object.train()

    if user_variables.get("whether_to_run_best_model"):
        test_object.run_best_model(number_of_games=user_variables.get("number_of_games"),whether_to_render=user_variables.get("whether_to_render"))

    test_object.env.close()


if __name__ == '__main__':
    # all possible changes are editable by updating user_variables
    user_variables = {
        # train and run at the end
        "whether_to_train": True, # Boolean to train 
        "whether_to_run_best_model": False, #Boolean to test the current best model
        "whether_to_render": False, # Boolean to render test
        # training settings
        "total_episodes": 50000, # Number of episodes to run
        "batch_size": 32,
        "episode_length": 251, # Depth of an episode (Hockey env has max depth 251)
        "exploration_noise": 0.01,
        "min_exploration_noise": 0.0001,
        "exploration_decay": 0.999999,
        "train_against_weak_n_number_of_times": 1000,
        # running settings
        "number_of_games": 100, # Number of games to test an episode's quality or to render - a high number is recommended since it effected my training a lot
        "opponents": [h_env.BasicOpponent(weak=False), h_env.BasicOpponent(weak=False)], # list of opponents
        # TD3 settings
        "discount": 0.999999,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        "hidden_dim_1": 256,
        "hidden_dim_2": 256,
        "learning_rate": 1e-6,
        "weight_decay": 1e-5,
        "grad_clip": 1.0, 
        "memory_limit": 100000 # memory limit to allocate for the task
    }
    main(user_variables)