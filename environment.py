from numpy.random import randint
from random import choices
from dataclasses import dataclass
import logging
import sys
from enum import Enum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler((stdout_handler))


class Colour(Enum):
    RED = 0
    BLACK = 1


class Card:
    def __init__(self, colour: Colour, value):
        self.value = value
        self.colour = colour


@dataclass()
class State:
    dealer_first_card: int  # always black
    player_sum: int
    terminal: bool = False


class Action(Enum):
    STICK = 0
    HIT = 1


class GameRound:
    """The simulated Easy 21 game environment"""

    def __init__(self, state: State, action) -> None:
        self.state: State = state
        self.action: Action = action

    def play_game(self):
        if self.action == Action.HIT:
            logger.debug("we hit")
            new_player_sum = self.hit(player_sum=self.state.player_sum)
            if new_player_sum > 21 or new_player_sum < 1:
                logger.debug("we went bust")
                new_terminal = True
                new_reward = -1
            else:
                new_terminal = False
                new_reward = 0

        elif self.action == Action.STICK:
            logger.debug("we stick")
            new_terminal = True
            new_player_sum = self.state.player_sum
            dealer_bust, dealer_sum = self.play_dealer()

            if dealer_bust:
                logger.debug("the dealer is bust")
                new_reward = 1
            else:
                logger.debug(f"we have: {new_player_sum}, dealer has {dealer_sum} ")
                if new_player_sum > dealer_sum:
                    new_reward = 1
                    logger.debug("we won")
                elif new_player_sum == dealer_sum:
                    logger.debug("we draw")
                    new_reward = 0
                elif new_player_sum < dealer_sum:
                    logger.debug("we lose")
                    new_reward = -1

        new_state = State(
            self.state.dealer_first_card,
            new_player_sum,
            new_terminal,
        )
        logger.debug(new_state)
        logger.debug(new_reward)

        return new_state, new_reward

    def play_dealer(self):
        dealer_sum = self.state.dealer_first_card
        while 1 <= dealer_sum < 17:
            logger.debug(f"dealer is hitting on hand {dealer_sum}")
            dealer_sum = self.hit(dealer_sum)
            logger.debug(f"now dealer has {dealer_sum}")
        if dealer_sum > 21 or dealer_sum < 1:
            dealer_bust = True
        else:
            dealer_bust = False
        return dealer_bust, dealer_sum

    @staticmethod
    def draw_next_card():
        value = randint(1, 11)
        colour = choices(population=[Colour.RED, Colour.BLACK], weights=[1 / 3, 2 / 3])[
            0
        ]
        return Card(colour, value)

    def hit(self, player_sum: int):
        new_card = self.draw_next_card()
        if new_card.colour == Colour.BLACK:
            player_sum += new_card.value
        elif new_card.colour == Colour.RED:
            player_sum -= new_card.value
        logger.debug(
            f"with new card {new_card.colour, new_card.value} player sum is {player_sum}"
        )
        return player_sum


class Environment:
    def __init__(self, start_state=None):
        if start_state == None:
            self.start_state = self.get_start_state()
        else:
            self.start_state = start_state

    def step(self, state: State, action: str):
        game = GameRound(state, action)
        new_state, reward = game.play_game()

        return new_state, reward

    @staticmethod
    def get_start_state():
        start_state = State(
            dealer_first_card=randint(1, 11), player_sum=randint(1, 11), terminal=False
        )
        return start_state


if __name__ == "__main__":
    env = Environment()
    new_state = env.step(env.start_state, Action.STICK)
