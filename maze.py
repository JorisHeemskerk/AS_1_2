import numpy as np

from action import Action
from state import State


class Maze:
    """
    Maze class.

    A maze is defined by a set of states, in the shape of a grid.
    @see state.py

    A maze can be initialized with a set of rewards, which is the score
    any agent gets for entering the state on the given coordinate.
    """

    def __init__(
        self, 
        grid_shape: tuple[int, int], 
        rewards: np.ndarray
    )-> None:
        """
        @var $states
        **np.ndarray** Numpy matrix with all the states.
        """
        if grid_shape != rewards.shape:
            raise AttributeError(
                f"`rewards` does not have the correct shape."
                f" Expected {grid_shape}, got {rewards.shape}."
        )

        # instantiate `states`, given the provided `grid_shape`
        self.states = np.empty(shape=grid_shape, dtype=object)
        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                self.states[x,y] = State((x,y), rewards[x,y], False)

    def __getitem__(self, coordinate: tuple[int, int])-> State:
        """
        Indexing dunder method. 

        This method makes it possible for the maze class 
        to be indexable, using a tuple with an x and y coordinate.

        @param coordinate: coordinate to get state from

        @return State with state on given coordinates
        """
        try:
            return self.states[coordinate]
        except IndexError:
            raise IndexError(
                f"Index out of range."
                f" Tried accessing index {coordinate} from `self.states, "
                f"which has shape of {self.states.shape}"
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def set_rewards(self, rewards: np.ndarray)-> None:
        """
        Setter for rewards in states.

        NOTE: `rewards` must match shape of `self.states`

        @param rewards: matrix with rewards corresponding to states.
        """
        if self.states.shape != rewards.shape:
            raise AttributeError(
                f"`rewards` does not have the correct shape."
                f" Expected {self.states.shape}, got {rewards.shape}."
        )
        for x in range(self.states.shape[0]):
            for y in range(self.states.shape[1]):
                self.states.shape[x,y].reward = rewards[x,y]

    def set_terminal(self, coordinate: tuple[int, int])-> None:
        """
        Setter for terminal state.

        This method lets you set a terminal state in the maze.

        @param coordinate: Coordinate of terminal state to be set.
        """
        try:
            self.states[coordinate].is_terminal = True
        except IndexError:
            raise IndexError(
                f"Index out of range."
                f" Tried accessing index {coordinate} from `self.states, "
                f"which has shape of {self.states.shape}"
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def step(
        self, 
        start_coordinate: tuple[int, int], 
        action: Action
    )-> tuple[int, int]:
        """
        Step function for Maze.

        This function checks if a certain action, from a given state,
        is valid. If so, the new coordinate is returned.
        If not, a corresponding error will be returned.

        @param start_coordinate: coordinate from where the `action`
            will be performed.
        @param action: actioin to take at `start_coordinate`

        @return tuple[int, int] with end coordinate
        """
        
        new_coordinate = tuple(map(sum, zip(start_coordinate, action.value)))

        # `new_coordinate` may not contain negative values
        if new_coordinate[0] < 0 or new_coordinate[1] < 0:
             raise IndexError(
                f"This action is invalid. The new coordinate would be "
                f"{new_coordinate}, which is out of range in a grid with shape"
                f" {self.states.shape}"
            )
        # `new_coordinate` must be within the bounds of the grid
        try:
            self.states[new_coordinate]
        except IndexError:
            raise IndexError(
                f"This action is invalid. The new coordinate would be "
                f"{new_coordinate}, which is out of range in a grid with shape"
                f" {self.states.shape}"
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        return new_coordinate

    def __str__(
        self, 
        agent_coordinate: tuple[int, int]=None, 
        agent_colour: str="\033[93m"
        )-> str:
        """
        Stringify current maze.

        Example:\n
        
        \nMaze class, with following grid:
        \n┌──────────────────┬──────────────────┬──────────────────┐
        \n│ ( 0,3 ), r =  0  │ ( 1,3 ), r = 90  │ ( 2,3 ), r = 22  │
        \n├──────────────────┼──────────────────┼──────────────────┤
        \n│ ( 0,2 ), r = 57  │ ( 1,2 ), r = 28  │ ( 2,2 ), r = 35  │
        \n├──────────────────┼──────────────────┼──────────────────┤
        \n│ ( 0,1 ), r = 25  │ ( 1,1 ), r = 23  │ ( 2,1 ), r = 91  │
        \n├──────────────────┼──────────────────┼──────────────────┤
        \n│ ( 0,1 ), r = 25  │ ( 1,1 ), r = 23  │ ( 2,1 ), r = 91  │
        \n└──────────────────┴──────────────────┴──────────────────┘

        @return str with stringified current maze
        """
        # base case for the horizontal lines
        deviding_line = f"{('─' * 18 + '┼') * (self.states.shape[0] - 1)}"\
        f"{'─' * 18}"

        output = f"Maze class, with following grid:"\
        f"\n┌{deviding_line.replace('┼', '┬')}┐\n│ "

        # transform and reverse matrix, 
        # such that (0, 0) starts in the bottom left
        reversed_transformed_states = self.states.T[::-1]
        for y, row in enumerate(reversed_transformed_states[:-1]):
            for x, state in enumerate(row.tolist()):
                if (x, self.states.shape[1]-1-y) == agent_coordinate:
                    output += state.__str__(agent_colour) + " │ "
                else:
                    output += str(state) + " │ "
            output += f"\n├{deviding_line}┤\n│ "

        # different formatting for last line
        for x, state in enumerate(row.tolist()):
            if (x, self.states.shape[1]-1-\
            reversed_transformed_states.shape[0]) == agent_coordinate:
                output += state.__str__(agent_colour) + " │ "
            else:
                output += str(state) + " │ "
        output += f"\n└{deviding_line.replace('┼', '┴')}┘"
        return output
