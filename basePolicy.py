import random

from action import Action
from state import State


class BasePolicy:
    """
    BasePolicy
    
    Base policy class with random behavior.
    This policy works as follows:
    - select select a random action and return it.
    """

    def __init__(self)-> None:
        """
        Initializer for BasePolicy

        This class has no member variables, 
        meaning the initializer does nothing.
        """
        pass

    def select_action(self, state: State)-> Action:
        """
        Select action based on current policy.
        
        This policy works as follows:
        - select select a random action and return it.

        @param state: Current State to perform action in.

        @return Action with Action to perform.
        """
        index = random.randrange(0, 4)
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT][index]
    
    def __str__(self) -> str:
        """
        stringify policy by just returning the class name.

        @return str with class name
        """
        return self.__class__.__name__
