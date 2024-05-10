import numpy as np

from baseAgent import BaseAgent
from basePolicy import BasePolicy
from optimalPolicy import OptimalPolicy
from maze import Maze

def main()-> None:
    maze_shape = (4,4)

    rewards = np.array([
        [10,  -1,  -1,  -1],
        [-2,  -1,  -1,  -1],
        [-1,  -1, -10,  -1],
        [-1,  -1, -10,  40],
    ], dtype=int) # reward matrix from assignment

    # use line below if you want a random reward maze instead
    # rewards = np.random.randint(0, 100, size=maze_shape)

    maze = Maze(maze_shape, rewards)
    maze.set_terminal((0,0))
    maze.set_terminal((3,3))

    agent = BaseAgent(maze, BasePolicy(), (2,0))


    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    print(f"\033[32m{'─'*49}\n\t\tOptimizing Policy\n{'─'*49}\033[0m")
    policy = OptimalPolicy(maze, 0.01, 1, True)


    print(f"\033[32m{'─'*45}\n\t\tAgent actions\n{'─'*45}\033[0m")
    # Assign policy to agent
    agent.policy = policy

    print(agent)
    # keep going until terminate state is reached
    while not maze[agent.current_coordinate].is_terminal:
        agent.act(True)

if __name__ == "__main__":
    main()
