import logging
import sys
import numpy as np

from q1_environment import Action, Game, State
from utils import Visuals

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler((stdout_handler))
np.random.seed(seed=32)


class SarsaAgent:
    """
    Agent that will play Sarsa(lamda)
    across many episodes.
    Uses linear approximator theta.
    """

    N0 = 100

    def __init__(self, lamda):
        self.win_rate = 0
        self.lamda = lamda
        self.epsilon = 0.05
        self.alpha = 0.01
        self.theta = np.random.randn(36) * 0.1  # our approximator we do grad desc on

    def action_from_policy(self, state: State):
        """
        Implements epsilon greedy policy.
        With P = 1-eps chooses best action, with P = eps chooses random action.
        """
        p = np.random.random()
        if p <= 1 - self.epsilon:  #  with prob 1-eps
            action = (
                Action.STICK.value
                if self.get_action_value(state, Action.STICK)
                > self.get_action_value(state, Action.HIT)
                else Action.HIT.value
            )
        else:
            action = np.random.randint(2)

        return Action(action)

    def get_action_value(self, state: State, action: Action):
        return np.dot(get_feature(state, action), self.theta)

    def get_full_action_values(self):
        action_value_function = np.zeros(
            (10 + 1, 21 + 1, 2)
        )  # dealer showing x player sum x action
        for dealer_first_card in range(1, 10 + 1):
            for hand_value in range(1, 21 + 1):
                for action in [Action.HIT, Action.STICK]:
                    s = State(dealer_first_card, hand_value)
                    action_value_function[
                        dealer_first_card, hand_value, action.value
                    ] = self.get_action_value(s, action)
        return action_value_function


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
                - update theta and E for all states and actions
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
        self.eligibility_trace = np.zeros(36)  # action-value approx

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
                + self.agent.get_action_value(self.new_state, self.new_action)
                - self.agent.get_action_value(self.old_state, self.old_action)
            )
        else:  # terminal new state, new Q = 0
            td_error = reward - self.agent.get_action_value(
                self.old_state, self.old_action
            )

        # bump E eligibility trace for old state by feaute vector
        self.eligibility_trace += get_feature(self.old_state, self.old_action)
        # update theta
        self.agent.theta += self.agent.alpha * td_error * self.eligibility_trace
        # update E
        self.eligibility_trace = self.agent.lamda * self.eligibility_trace


class Train:
    """This class is training the agent for many episodes."""

    def __init__(self, episode_count=None, lamda=None) -> None:
        self.episode_count = episode_count
        self.agent = SarsaAgent(lamda)
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
                    f"linear_results_{self.lamda}/linear_lamda_{current_episode_count}",
                    self.agent.get_full_action_values(),
                )
                Visuals.plot_value(
                    f"linear_results_{self.lamda}/linear_lamda_{current_episode_count}.npy",
                    current_episode_count,
                    self.lamda,
                )
        np.save(f"linear_results_{self.lamda}/linear_win_rate", self.win_rates)
        Visuals.plot_win_rates(
            f"linear_results_{self.lamda}/linear_win_rate.npy", self.lamda
        )
        Visuals.make_gif(f"linear_results_{self.lamda}", f"Linear_control_{self.lamda}")

    def run_sarsa_for_assignment(self):
        """Sarsa(lamda) for different lamda values, ran for 1000 iterations"""
        final_mse = {}

        with open("mc_control_results/MC_Control_1000000.npy", "rb") as f:
            true_q = np.load(f)
        for lamda in np.linspace(0, 1, 11):
            self.agent = SarsaAgent(lamda)
            self.lamda = lamda
            if lamda in {0, 1}:
                lamda = int(lamda)
                mean_squared_errors = np.zeros((10000))
            for current_episode_count in range(10000):
                episode = Episode(self.agent)
                episode.run_episode()
                if lamda in {0, 1}:
                    mean_squared_errors[current_episode_count] = mean_sqr(
                        true_q, self.agent.get_full_action_values()
                    )

            final_mse[lamda] = mean_sqr(true_q, self.agent.get_full_action_values())
            if lamda in {0, 1}:
                np.save(
                    f"linear_approx_assignment/linear_approx_learning_{lamda}",
                    mean_squared_errors,
                )
                Visuals.plot_mse_learning(
                    f"linear_approx_assignment/linear_approx_learning_{lamda}.npy",
                    lamda,
                )
            Visuals.plot_mse_for_lamdas(final_mse, "linear_approx_assignment")


def mean_sqr(q1, q2):
    return np.sum(np.square(q1 - q2)) / q1.size


def get_feature(state: State, action: Action):
    """Returns the feature vector of the state"""
    phi = np.zeros((3, 6, 2))
    dealer_features = [[1, 4], [4, 7], [7, 10]]
    dealer_indexes = np.where(
        np.array([x[0] <= state.dealer_first_card <= x[1] for x in dealer_features])
        == 1
    )[0]
    agent_features = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]

    agent_indexes = np.where(
        np.array([x[0] <= state.hand_value <= x[1] for x in agent_features]) == 1
    )[0]

    action_idx = action.value

    for dealer_idx in dealer_indexes:
        for agent_idx in agent_indexes:
            phi[dealer_idx, agent_idx, action_idx] = 1

    phi = phi.flatten()
    return phi


if __name__ == "__main__":
    # trainer = Train(episode_count=1000000, lamda=0)
    # trainer.run_sarsa_for_gif()
    trainer = Train()
    trainer.run_sarsa_for_assignment()
