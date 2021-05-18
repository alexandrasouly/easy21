import logging
from typing import Tuple
from environment import Action, Environment, State
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
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
            (10 + 1, 21 + 1, 2)
        )  # dealer showing x player sum x action
        self.state_count = np.zeros(
            (10 + 1, 21 + 1)
        )  # dealer showing x player sum x action
        self.state_action_count = np.zeros(
            (10 + 1, 21 + 1, 2)
        )  # dealer showing x player sum x action
        self.win_rate = 0

    def eps(self, state: State):
        return self.N0 / (
            self.N0 + self.state_count[state.dealer_first_card, state.player_sum]
        )

    def alpha(self, state: State, action: Action):
        return (
            1
            / self.state_action_count[
                state.dealer_first_card, state.player_sum, action.value
            ]
        )

    def action_from_policy(self, state: State):  # this is the policy, based on Q(s,a)
        """Implements epsilon greedy policy."""
        p = np.random.random()
        if p <= 1 - self.eps(state):  #  with prob 1-eps
            action = np.argmax(
                self.action_value_function[state.dealer_first_card, state.player_sum, :]
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
        self.win = False

    def run_episode(self):  # run an episode based on policy
        while self.state.terminal == False:
            action_int = self.agent.action_from_policy(self.state)
            new_state, reward = self.env.step(self.state, action=Action(action_int))
            self.visited.append((self.state, Action(action_int), reward))
            self.state = new_state
        self.update_policy()
        if reward == 1:
            self.win = True

    def update_policy(self):
        for state, action, _ in self.visited:
            self.agent.state_action_count[
                state.dealer_first_card, state.player_sum, action.value
            ] += 1
            self.agent.state_count[state.dealer_first_card, state.player_sum] += 1
            final_reward = self.visited[-1][2]
            error = (
                final_reward
                - self.agent.action_value_function[
                    state.dealer_first_card, state.player_sum, action.value
                ]
            )
            self.agent.action_value_function[
                state.dealer_first_card, state.player_sum, action.value
            ] += (self.agent.alpha(state, action) * error)


class Train:
    def __init__(self, episode_count) -> None:
        self.episode_count = episode_count
        self.agent = MCControlAgent()
        self.win_rates = []
        self.wins = 0

    def run_mc(self):
        for current_episode_count in range(1, self.episode_count + 1):
            episode = Episode(self.agent)
            episode.run_episode()
            if episode.win:
                self.wins += 1
                win_rate = self.wins / (current_episode_count)

            if current_episode_count % 10000 == 0:
                logger.info(
                    f" After episode {current_episode_count}, the win rate is: {win_rate}"
                )
                self.win_rates.append(win_rate)


def plot_value(action_value_file):
    with open(action_value_file, "rb") as f:
        action_value = np.load(f)

    value = np.max(action_value, axis=2)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection="3d")
    x_range = np.arange(1, action_value.shape[0])
    y_range = np.arange(1, action_value.shape[1])
    X, Y = np.meshgrid(x_range, y_range)
    Z = value[X, Y]
    ax.set_xlabel("Dealer First card")
    ax.set_ylabel("Player Sum")
    ax.set_zlabel("Value")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    plt.savefig("Value_fn.png")


if __name__ == "__main__":
    trainer = Train(3)
    trainer.run_mc()

    np.save("MC_Control_action_value", trainer.agent.action_value_function)
    np.save("MC_Control_win_rate", trainer.win_rates)

    plt.plot(trainer.win_rates)
    plt.xlabel("Thousand episode played")
    plt.ylabel("Win Rate")
    plt.savefig("MC_control.png")

    plot_value("MC_Control_action_value.npy")
