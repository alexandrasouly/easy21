import logging
import sys
from dataclasses import dataclass
from enum import Enum
from random import choices
from typing import Tuple

from numpy.random import randint

# Setting up logging to stdout INFO level logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler((stdout_handler))


class Colour(Enum):
    """Colour enums for the playing cards"""

    RED = 0
    BLACK = 1


class Card:
    """Card obejcts having a colour enum and an integer value - no face cards."""

    def __init__(self, colour: Colour, value):
        self.value = value
        self.colour = colour


class Action(Enum):
    """The enums representing the two actions we can take."""

    STICK = 0
    HIT = 1


@dataclass()
class State:
    """This is how we are stroing the state in a round of easy21."""

    dealer_first_card: int  # always black, so +
    hand_value: int
    terminal: bool = False


class GameRound:
    """A round (one action by the player) of Easy 21 game."""

    def __init__(self, state: State, action) -> None:
        """Starting state and the action the player takes."""
        self.state: State = state
        self.action: Action = action

    def play_round(self) -> Tuple[int, State]:
        """The player takes the action, we return the reward and get to a new state"""

        if self.action == Action.HIT:
            logger.debug("Player hits")
            new_hand_value = self.hit(hand_value=self.state.hand_value)
            if new_hand_value > 21 or new_hand_value < 1:
                logger.debug("Player went bust")
                new_terminal = True
                new_reward = -1
            else:
                new_terminal = False
                new_reward = 0

        elif self.action == Action.STICK:
            logger.debug("Player sticks.")
            new_terminal = True
            new_hand_value = self.state.hand_value
            dealer_bust, dealer_sum = self.play_dealer()

            if dealer_bust:
                logger.debug("The dealer went bust.")
                new_reward = 1
            else:
                logger.debug(f"Player has: {new_hand_value}, dealer has {dealer_sum}.")
                if new_hand_value > dealer_sum:
                    new_reward = 1
                    logger.debug("Player has won.")
                elif new_hand_value == dealer_sum:
                    logger.debug("Draw!")
                    new_reward = 0
                elif new_hand_value < dealer_sum:
                    logger.debug("Player has lost.")
                    new_reward = -1

        new_state = State(
            self.state.dealer_first_card,
            new_hand_value,
            new_terminal,
        )

        return new_state, new_reward

    def play_dealer(self) -> Tuple[bool, int]:
        """The policy of the dealer: draw until 17 or higher, then stick."""
        dealer_sum = self.state.dealer_first_card
        while 1 <= dealer_sum < 17:
            logger.debug(f"The dealer is hitting on {dealer_sum}")
            dealer_sum = self.hit(dealer_sum)
            logger.debug(f"Now dealer has {dealer_sum}.")
        if dealer_sum > 21 or dealer_sum < 1:
            dealer_bust = True
        else:
            dealer_bust = False
        return dealer_bust, dealer_sum

    @staticmethod
    def draw_next_card() -> Card:
        """Drawing a card out of our infinite deck that has 1/3 red, 2/3 black."""
        value = randint(1, 11)
        colour = choices(population=[Colour.RED, Colour.BLACK], weights=[1 / 3, 2 / 3])[
            0
        ]
        return Card(colour, value)

    def hit(self, hand_value: int) -> int:
        """Play the hit action on a current hand value."""
        new_card = self.draw_next_card()
        if new_card.colour == Colour.BLACK:
            hand_value += new_card.value
        elif new_card.colour == Colour.RED:
            hand_value -= new_card.value
        return hand_value


class Game:
    """A class to use play the game, implementing the API for the agents to use."""

    def __init__(self, start_state=None):
        """We initialise a game, to a random start state unless specified."""
        if start_state == None:
            self.start_state = self._get_start_state()
        else:
            self.start_state = start_state

    def step(self, state: State, action: str):
        """Main game API defined by the assignment, our agents will use this."""
        round = GameRound(state, action)
        new_state, reward = round.play_round()

        return new_state, reward

    @staticmethod
    def _get_start_state() -> State:
        """Get the start state by drawing a random black card for player and dealer."""
        start_state = State(
            dealer_first_card=randint(1, 11), hand_value=randint(1, 11), terminal=False
        )
        return start_state
