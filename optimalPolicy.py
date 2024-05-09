from dataclasses import dataclass
import inspect
import numpy as np
from typing import Annotated, get_type_hints

from action import Action
from basePolicy import BasePolicy
from maze import Maze
from state import State


def check_annotated(func)-> None:
    """
    Checker wrapper function to force type annotations.

    @param func: function to wrap around
    """
    hints = get_type_hints(func, include_extras=True)
    spec = inspect.getfullargspec(func)
    def wrapper(*args, **kwargs):
        for idx, arg_name in enumerate(spec[0]):
            hint = hints.get(arg_name)
            validators = getattr(hint, '__metadata__', None)
            if not validators:
                continue
            for validator in validators:
                validator.validate_value(args[idx])
        return func(*args, **kwargs)
    return wrapper


@dataclass
class FloatRange:
    """
    FloatRange class

    This class is used to limit the values for floats in certain parameters
    """
    min: float
    max: float

    def validate_value(self, x: float)-> None:
        """
        Validation function for value using `FloatRange`

        provided `x` should be between self.min and self.max (inclusive)

        @param x: float to validate
        """
        if not (self.min <= x <= self.max):
            raise ValueError(f'{x} must be in range [{self.min}, {self.max}].')
        

class OptimalPolicy(BasePolicy):
    """
    OptimalPolicy
    
    Base policy class with optimal behavior, given an MDP.
    This policy works as follows:
    - select select the best action, given the MDP, and return it.
    """

    @check_annotated
    def __init__(
        self, 
        maze: Maze, 
        threshold: Annotated[float, FloatRange(0.0, float("inf"))],
        discount: Annotated[float, FloatRange(0.0, 1.0)]=0.9,
        visualise: bool=False
    )-> None:
        """
        @var $maze
        **Maze** with MDP information

        @var $actions
        **dict[state : action]** 
        dictionary with states mapping to optimal actions.  
        """
        super().__init__()

        self.maze = maze
        self.actions = self._determine_optimal_policy(
            threshold, 
            discount, 
            visualise
        )

    @check_annotated
    def _determine_optimal_policy(
        self, 
        threshold: Annotated[float, FloatRange(0.0, float("inf"))],
        discount: Annotated[float, FloatRange(0.0, 1.0)]=0.9,
        visualise: bool=False
    )-> dict[State : Action]:
        """
        Determine optimal policy for given `self.maze`.

        This function performs the bellman function on the MDP
        in order to calculate the optimal policy 
        (using the value iteration)
        
        @param threshold: float greater than 0.0 with threshold for
        when to stop converging
        @param discount: discount for future values/states
        @param visualise: print value matrix after each iteration
        if true

        @return dict[State : Action] with optimal policy
        """
        actions = {state: None for state in self.maze.states.flatten()}
        previous_values = {state: 0 for state in self.maze.states.flatten()}
        delta = float("inf")
        iteration = 0
        
        while delta >= threshold:
            delta = 0
            new_values = previous_values.copy()
            for state in self.maze.states.flatten():
                # terminal states have a value of 0 
                # and policy of no action
                if state.is_terminal:
                    new_values[state] = 0
                    continue

                # find possible actions
                possible_actions = {}
                for action in [
                    Action.UP, 
                    Action.DOWN, 
                    Action.LEFT, 
                    Action.RIGHT
                ]:
                    try:
                        destination = self.maze.step(
                            state.position, action
                        )
                        possible_actions[action] = self.maze[destination]
                    except:
                        continue

                # determine new value using $V(s) \leftarrow 
                # {max}_a \sum_{s',r}^{} 
                # p(s', r | s, a) [r + \gamma V(s')]$
                expected_values = [
                    (
                        action, 
                        state.reward + discount * previous_values[state]
                    ) for action, state in possible_actions.items()
                ]
                # extract max value
                max_value = max(
                    [expected_value[1] for expected_value in expected_values]
                )
                # save best action for eventual 
                for action, value in expected_values:
                    if value == max_value:
                        actions[state] = action
                        break
                # save maximal value
                new_values[state] = max_value

                # calculate new delta
                delta = max(
                    [delta, previous_values[state] - new_values[state]]
                )
            iteration += 1 
            previous_values = new_values

            if visualise:
                print(
                    f"Values for current iteration ({iteration}),",
                    f"with current delta of {delta}:"
                )
                print(self.values_in_maze_to_str(new_values))
        return actions

    def values_in_maze_to_str(self, values: dict[State : float])-> str:
        """
        Stringify values into maze matrix.

        Example:\n
        
        \n Values for current iteration (3), with current delta of 0.99:
        \n┌─────────────────────────┬─────────────────────────┐
        \n│ ( 0,2 ), v =  8.900000  │ ( 1,2 ), v = 37.214000  │
        \n├─────────────────────────┼─────────────────────────┤
        \n│ ( 0,1 ), v = 10.000000  │ ( 1,1 ), v =  8.900000  │
        \n├─────────────────────────┼─────────────────────────┤
        \n│ ( 0,0 ), v =  0.000000  │ ( 1,0 ), v = 10.000000  │
        \n└─────────────────────────┴─────────────────────────┘

        @return str with stringified values into maze matrix
        """
        # base case for the horizontal lines
        deviding_line = \
            f"{('─' * 25 + '┼') * (self.maze.states.shape[0] - 1)}"\
            f"{'─' * 25}"

        output = f"┌{deviding_line.replace('┼', '┬')}┐\n│ "

        # transform and reverse matrix, 
        # such that (0, 0) starts in the bottom left
        reversed_transformed_states = self.maze.states.T[::-1]
        for row in reversed_transformed_states[:-1]:
            for state in row.tolist():
                output += str(state).split('r')[0] + \
                    "v = {:^10.6f}\033[0m │ ".format(values[state])
            output += f"\n├{deviding_line}┤\n│ "

        # different formatting for last line
        for state in reversed_transformed_states[-1].tolist():
            output += str(state).split('r')[0] + \
                "v = {:^10.6f}\033[0m │ ".format(values[state])
        output += f"\n└{deviding_line.replace('┼', '┴')}┘"
        return output

    def select_action(self, state: State)-> Action:
        """
        Select action based on current policy.
        
        This policy works as follows:
        - select select the best action, given the MDP, and return it.

        @param state: Current State to perform action in.

        @return Action with Action to perform.
        """
        return self.actions[state]
    