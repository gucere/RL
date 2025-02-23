from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np
from emre_version.TD3_alternative_new import TD3
import torch


from comprl.client import Agent, launch_client


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class TD3HockeyAgent(Agent):
    """A hockey agent that uses a trained TD3 model."""

    def __init__(self, model_path_actor: str, model_path_critic: str) -> None:
        super().__init__()

        # Initialize environment to get action space details
        self.env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        # Load TD3 agent
        self.agent = TD3(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=1,
            hidden_dim_1=256,
            hidden_dim_2=256,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            weight_decay=1e-5,
            grad_clip=1.0
        )

        # Load trained weights
        self.agent.actor.load_state_dict(torch.load(model_path_actor, map_location="cpu"))
        self.agent.critic.load_state_dict(torch.load(model_path_critic, map_location="cpu"))
        self.agent.actor.eval()  # Set to evaluation mode
        self.agent.critic.eval()

        print("TD3 Agent Loaded Successfully!")

    def get_step(self, observation: list[float]) -> list[float]:
        """Selects an action using the trained TD3 model."""
        action = self.agent.select_action(observation).squeeze().tolist()  # Get action
        action1 = action[:4]
        return action1  # Return action in list format

    def on_start_game(self, game_id) -> None:
        #game_id = uuid.UUID(int=int.from_bytes(game_id))
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder="little"))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )




# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "td3"],
        default="weak",
        help="Which agent to use.",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "td3":
        return TD3HockeyAgent(model_path_actor="C:/Users/emreg/Downloads/RL/models/best_actor.pth", model_path_critic="C:/Users/emreg/Downloads/RL/models/best_critic.pth")
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
