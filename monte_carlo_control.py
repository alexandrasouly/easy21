from environment import Environment, State
import numpy as np


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
    def __init__(self):
        self.env = Environment()
        self.action_value_function = np.zeros(
            (10, 21, 2)
        )  # dealer showing x player sum x action
        self.state_count = np.zeros(
            (10, 21)
        )  # dealer showing x player sum x action
        self.state_action_count = np.zeros(
            (10, 21, 2)
        )  # dealer showing x player sum x action

        self.epsilon = 

    def choose_action():  # this is the policy, based on Q(s,a)
        p = np.random
        pass

    def run_episode():  # run an episode based on policy
        pass

    def update_policy():
        #update all time varying constants
        # update Q
        pass


if __name__ == "__main__":
    env = Environment()
    start_state = State(20, 20)
    new_state = env.step(start_state, "hit")