import logging
from typing import Tuple
from environment import Action, Environment, State
import numpy as np
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler((stdout_handler))


class MCControlAgent:
    """
    for episode in  many episodes:
        play the episode with our current policy
            initialise a state
            until in terminal state:
                act according to policy
                keep track of visited states, rewards
        update policy: Q, N(s,a), N(s), G, eps
    """

    N0 = 100

    def __init__(self):
        self.action_value_function = np.zeros(
            (10, 21, 2)
        )  # dealer showing x player sum x action
        self.state_count = np.zeros((10, 21))  # dealer showing x player sum x action
        self.state_action_count = np.zeros(
            (10, 21, 2)
        )  # dealer showing x player sum x action
        self.win_rate = 0

    def action_from_policy(self):  # this is the policy, based on Q(s,a)
        """Implements epsilon greedy policy."""
        p = np.random.random()
        if p <= 1 - self.epsilon:  #  with prob 1-eps
            action = np.argmax(
                self.action_value_function[
                    self.state.dealer_first_card, self.state.player_sum, :
                ]
            )
        else:
            action = np.random.randint(2)

        return Action(action)


class Episode:
    def __init__(self, agent: MCControlAgent) -> None:
        self.env = Environment()
        self.agent = agent
        self.state = self.env.start_state
        self.visited: Tuple[State, Action, int] = []

    def run_episode(self):  # run an episode based on policy
        while self.state.terminal == False:
            action = self.agent.action_from_policy()
            new_state, reward = self.env.step(self.state, action=action)
            self.visited.append((self.state, action, reward))
            self.state = new_state
        self.update_policy()

    def update_policy():
        # update all time varying constants
        # update Q
        pass


class Train:
    def __init__(self, episode_count) -> None:
        self.episode_count = episode_count
        self.agent = MCControlAgent()

    def run_mc(self):
        for episode_count in range(self.episode_count):
            episode = Episode(self.agent)
            episode.run_episode()
            if episode_count % 100:
                logger.info(
                    f" After episode {episode_count}, the win rate is: {self.agent.win_rate}"
                )


if __name__ == "__main__":
    env = Environment()
    start_state = State(20, 20)
    new_state = env.step(start_state, "hit")
