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


class SarsaAgent:
    """
    Agent that will play Sarsa(lamda)
    across many episodes.

    """

    N0 = 100

    def __init__(self, lamda):
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
        self.lamda = lamda

    def eps(self, state: State):
        """Epsilon for the epsilon-greedy policy: eps = N0/(N0 + N(s))"""
        return self.N0 / (
            self.N0 + self.state_count[state.dealer_first_card, state.hand_value]
        )

    def alpha(self, state: State, action: Action):
        """Alpha for the Q update step size; α = 1/N(s, a)"""
        return (
            1
            / self.state_action_count[
                state.dealer_first_card, state.hand_value, action.value
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
                self.action_value_function[state.dealer_first_card, state.hand_value, :]
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
            - initialise a state and empty eligibiliy trace
            - until in terminal state:
                - act according to policy
                - get td-error
                - bump E, N(s,a), N(s), for current pre-step state and action
                - update Q and E for all states and actions
    """

    def __init__(self, agent: SarsaAgent) -> None:
        self.env = Game()
        self.agent = agent
        # starting state S
        self.old_state = self.env.start_state
        # starting action A
        self.old_action = self.agent.action_from_policy(self.old_state)
        # new state after step S'
        self.new_state = None
        # new action A' from S'
        self.new_action = None
        self.win = False
        self.eligibility_trace = np.zeros(
            (10 + 1, 21 + 1, 2)  # dealer showing x player sum x action
        )

    def run_episode(self):
        """Run the episode"""

        while self.old_state.terminal == False:
            # play round with action
            self.new_state, old_reward = self.env.step(
                self.old_state, action=Action(self.old_action)
            )
            # get new action
            if not self.new_state.terminal:
                self.new_action = self.agent.action_from_policy(self.new_state)
            else:
                self.new_action = None
            self.update(old_reward)
            self.old_state = self.new_state
            self.old_action = self.new_action

        if old_reward == 1:
            self.win = True  # for win_rate calculations

    def update(self, reward):
        """Do all the necessary updates after a single step"""
        # the following are for most recent played state-action pair:
        # calculate TD error
        if self.new_action is not None:  # non-terminal new state state
            td_error = (
                reward
                + self.agent.action_value_function[
                    self.new_state.dealer_first_card,
                    self.new_state.hand_value,
                    self.new_action.value,
                ]
                - self.agent.action_value_function[
                    self.old_state.dealer_first_card,
                    self.old_state.hand_value,
                    self.old_action.value,
                ]
            )
        else:  # terminal new state, new Q = 0
            td_error = (
                reward
                - self.agent.action_value_function[
                    self.old_state.dealer_first_card,
                    self.old_state.hand_value,
                    self.old_action.value,
                ]
            )

        # bump E eligibility trace for old state by 1
        self.eligibility_trace[
            self.old_state.dealer_first_card,
            self.old_state.hand_value,
            self.old_action.value,
        ] += 1
        # bump N(s,a) state-action visit count
        self.agent.state_action_count[
            self.old_state.dealer_first_card,
            self.old_state.hand_value,
            self.old_action.value,
        ] += 1
        # bump N(s) state visit count
        self.agent.state_count[
            self.old_state.dealer_first_card,
            self.old_state.hand_value,
        ] += 1
        # the following are for all state-action pairs:
        # update Q
        self.agent.action_value_function += (
            self.agent.alpha(self.old_state, self.old_action)
            * td_error
            * self.eligibility_trace
        )
        # update E
        self.eligibility_trace = self.agent.lamda * self.eligibility_trace


class Train:
    """This class is training the agent for many episodes."""

    def __init__(self, episode_count, lamda) -> None:
        self.episode_count = episode_count
        self.agent = SarsaAgent(lamda)
        self.mse = []
        self.win_rates = []
        self.wins = 0
        self.lamda = lamda
        self.print_episodes = [  # the episodes we want to plot at the end
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

    def run_sarsa_for_gif(self):
        """Run the full sarsa algorithm for many episodes, similarly to MC."""
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
                    f"sarsa_results_{lamda}/sarsa_lamda_{current_episode_count}",
                    self.agent.action_value_function,
                )
                Visuals.plot_value(
                    f"sarsa_results_{lamda}/sarsa_lamda_{current_episode_count}.npy",
                    current_episode_count,
                    self.lamda,
                )

    # def run_sarsa_for_assignment(self):
    #     """Sarsa(lamda) for different lamda values, ran for 1000 iterations """
    #     for lamda


if __name__ == "__main__":
    lamda = 1
    trainer = Train(1000000, lamda)
    trainer.run_sarsa_for_gif()

    np.save(f"sarsa_results_{lamda}/sarsa_win_rate", trainer.win_rates)
    Visuals.plot_win_rates(f"sarsa_results_{lamda}/sarsa_win_rate.npy", lamda)

    Visuals.make_gif(f"sarsa_results_{lamda}", f"Sarsa_control_{lamda}")
