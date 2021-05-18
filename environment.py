from _typeshed import NoneType
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


class Game:
    """The simulated Easy 21 game environment"""

    def __init__(self, state: State, action) -> None:
        self.state: State = state
        self.action: Action = action
        self.new_terminal: bool = False
        self.new_reward: int = 0

    def play_game(self):
        if self.action == Action.HIT:
            logger.info("we hit")
            self.new_player_sum = self.hit(player_sum=self.state.player_sum)
            if self.new_player_sum > 21 or self.new_player_sum < 1:
                logger.info("we went bust")
                self.new_terminal = True
                self.new_reward = -1

        if self.action == Action.STICK:
            logger.info("we stick")
            self.new_terminal = True
            self.new_player_sum = self.state.player_sum

            self.play_dealer()
            if self.dealer_bust:
                logger.info("the dealer is bust")
                self.new_reward = 1
            else:
                logger.info(
                    f"we have: {self.new_player_sum}, dealer has {self.dealer_sum} "
                )
                if self.new_player_sum > self.dealer_sum:
                    self.new_reward = 1
                    logger.info("we won")
                elif self.new_player_sum == self.dealer_sum:
                    logger.info("we draw")
                    self.new_reward = 0
                elif self.new_player_sum < self.dealer_sum:
                    logger.info("we lose")
                    self.new_reward = -1

        self.new_state = State(
            self.state.dealer_first_card,
            self.new_player_sum,
            self.new_terminal,
        )
        logger.info(self.new_state)
        logger.info(self.new_reward)

    def play_dealer(self):
        self.dealer_sum = self.state.dealer_first_card
        while self.dealer_sum < 17:
            logger.info(f"dealer is hitting on hand {self.dealer_sum}")
            self.dealer_sum = self.hit(self.dealer_sum)
            logger.info(f"now dealer has {self.dealer_sum}")
        if self.dealer_sum > 21 or self.dealer_sum < 1:
            self.dealer_bust = True
        else:
            self.dealer_bust = False

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
        logger.info(
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
        game = Game(state, action)
        game.play_game()
        new_state = game.new_state
        reward = game.new_reward

        return new_state, reward

    @staticmethod
    def get_start_state():
        start_state = State(
            dealer_first_card=randint(1, 11), player_sum=randint(1, 11), terminal=False
        )
        return start_state


if __name__ == "__main__":
    env = Environment()
    new_state = env.step(Action.STICK)
