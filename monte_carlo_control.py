import logging
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from environment import Action, Game, State
from utils import Visuals

# Setting up logging to stdout INFO level logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler((stdout_handler))


class MCControlAgent:
    """
    Agent that will play MC Control with epsilon greedy policy improvements
    across many episodes.

    """

    N0 = 100

    def __init__(self):
        # Q(s,a)
        self.action_value_function = np.zeros(
            (10 + 1, 21 + 1, 2)
        )  # dealer showing x player sum x action

        # N(s)
        self.state_count = np.zeros(
            (10 + 1, 21 + 1)
        )  # dealer showing x player sum x action

        # N(s,a)
        self.state_action_count = np.zeros(
            (10 + 1, 21 + 1, 2)
        )  # dealer showing x player sum x action
        self.win_rate = 0

    def eps(self, state: State):
        """Epsilon for the epsilon-greedy policy: eps = N0/(N0 + N(s))"""
        return self.N0 / (
            self.N0 + self.state_count[state.dealer_first_card, state.player_sum]
        )

    def alpha(self, state: State, action: Action):
        """Alpha for the Q update step size; α = 1/N(s, a)"""
        return (
            1
            / self.state_action_count[
                state.dealer_first_card, state.player_sum, action.value
            ]
        )

    def action_from_policy(self, state: State):
        """
        Implements epsilon greedy policy.
        With P = 1-eps chooses best action, with P = eps chooses random action.
        """
        p = np.random.random()
        if p <= 1 - self.eps(state):  #  with prob 1-eps
            action = np.argmax(
                self.action_value_function[state.dealer_first_card, state.player_sum, :]
            )
        else:
            action = np.random.randint(2)

        return Action(action)


class Episode:
    """
    Play a whole episode of Easy 21 with our agent.
    The structure of the whole process is the following:
    - for episode in  many episodes:
        - play the episode with our current policy
            - initialise a state
            - until in terminal state:
                - act according to policy
                - keep track of visited states, rewards
        - update policy: Q, N(s,a), N(s), G, eps
    This class is dealing with acting according to policy and tracing visited states.
    """

    def __init__(self, agent: MCControlAgent) -> None:
        self.env = Game()
        self.agent = agent
        self.state = self.env.start_state
        self.visited: Tuple[State, Action, int] = []
        self.win = False

    def run_episode(self):
        """Run the episode and update policy at the end"""
        while self.state.terminal == False:
            # get action
            action_int = self.agent.action_from_policy(self.state)
            # play round with aciton
            new_state, reward = self.env.step(self.state, action=Action(action_int))
            # track visited states
            self.visited.append((self.state, Action(action_int), reward))
            self.state = new_state
        self.update_policy()
        if reward == 1:
            self.win = True  # for win_rate calculations

    def update_policy(self):
        """At the end of the episode, update counter and policy for visited states."""
        for state, action, _ in self.visited:
            self.agent.state_action_count[
                state.dealer_first_card, state.player_sum, action.value
            ] += 1
            self.agent.state_count[state.dealer_first_card, state.player_sum] += 1
            final_reward = self.visited[-1][2]  # G_t
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
    """This class is training the agent for many episodes."""

    def __init__(self, episode_count) -> None:
        self.episode_count = episode_count
        self.agent = MCControlAgent()
        self.win_rates = []
        self.wins = 0
        self.print_episodes = [  # the episdoes we want to plot at the end
            1,
            5,
            10,
            25,
            50,
            100,
            150,
            200,
            300,
            500,
            1000,
            2500,
            5000,
            7500,
            10000,
            50000,
            100000,
            200000,
            300000,
            400000,
            600000,
            800000,
            self.episode_count,
        ]

    def run_mc_for_gif(self):
        """Run the full MC Control algorithm for many episodes."""
        for current_episode_count in range(1, self.episode_count + 1):
            episode = Episode(self.agent)
            episode.run_episode()
            if episode.win:
                self.wins += 1
                win_rate = self.wins / (current_episode_count)

            # print progress during training
            if current_episode_count % 10000 == 0:
                logger.info(
                    f" After episode {current_episode_count}, the win rate is: {win_rate}"
                )
                self.win_rates.append(win_rate)

            # save values function for final gif
            if current_episode_count in self.print_episodes:
                np.save(
                    f"mc_control_results/MC_Control_{current_episode_count}",
                    self.agent.action_value_function,
                )
                Visuals.plot_value(
                    f"mc_control_results/MC_Control_{current_episode_count}.npy",
                    current_episode_count,
                )


if __name__ == "__main__":
    trainer = Train(1000000)
    trainer.run_mc_for_gif()

    np.save("mc_control_results/MC_Control_win_rate", trainer.win_rates)
    Visuals.plot_win_rates("mc_control_results/MC_Control_win_rate.npy")

    Visuals.make_gif(folder="mc_control_results", gif_name="MC_Control")
