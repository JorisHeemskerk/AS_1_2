import numpy as np

from baseAgent import BaseAgent
from basePolicy import BasePolicy
from maze import Maze

def main()-> None:
    maze_shape = (4,4)
    maze = Maze(maze_shape, np.random.randint(0, 100, size=maze_shape))
    maze.set_terminal((1,3))

    policy = BasePolicy()

    agent = BaseAgent(maze, policy, (0,0))

    for _ in range(100):
        agent.act(True)
        if maze[agent.current_coordinate].is_terminal:
            break


if __name__ == "__main__":
    main()