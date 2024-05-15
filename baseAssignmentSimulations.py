import numpy as np

from baseAgent import BaseAgent
from basePolicy import BasePolicy
from optimalPolicy import OptimalPolicy
from probabilityAgent import ProbabilityAgent
from stupidMaze import StupidMaze

def simulate_base_assignment_A()-> None:
    """
    Creates maze from assignment.
    Print said maze.
    """
    maze_shape = (4,4)
    rewards = np.array([
        [10,  -1,  -1,  -1],
        [-2,  -1,  -1,  -1],
        [-1,  -1, -10,  -1],
        [-1,  -1, -10,  40],
    ], dtype=int) # reward matrix from assignment
    maze = StupidMaze(maze_shape, rewards)
    maze.set_terminal((0,0))
    maze.set_terminal((3,3))   

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze)

def simulate_base_assignment_B()-> None:
    """
    Creates maze from assignment.
    Places agent in maze.
    Print maze with agent.
    Let agent walk random path until terminal state is reached.
    """
    maze_shape = (4,4)
    rewards = np.array([
        [10,  -1,  -1,  -1],
        [-2,  -1,  -1,  -1],
        [-1,  -1, -10,  -1],
        [-1,  -1, -10,  40],
    ], dtype=int) # reward matrix from assignment
    maze = StupidMaze(maze_shape, rewards)
    maze.set_terminal((0,0))
    maze.set_terminal((3,3))   

    agent = BaseAgent(maze, BasePolicy(), (2,0))

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    print(f"\033[32m{'─'*45}\n\t\tAgent actions\n{'─'*45}\033[0m")
    print(agent)
    # keep going until terminate state is reached
    while not maze[agent.current_coordinate].is_terminal:
        agent.act(True)

def simulate_base_assignment_C()-> None:
    """
    Creates maze from assignment.
    Places agent in maze.
    Print maze with agent.
    Perform value iteration.
    Extract optimal policy.
    Print both.
    Have agent perform this optimal policy in maze.
    """
    maze_shape = (4,4)
    rewards = np.array([
        [10,  -1,  -1,  -1],
        [-2,  -1,  -1,  -1],
        [-1,  -1, -10,  -1],
        [-1,  -1, -10,  40],
    ], dtype=int) # reward matrix from assignment
    maze = StupidMaze(maze_shape, rewards)
    maze.set_terminal((0,0))
    maze.set_terminal((3,3))   

    agent = BaseAgent(maze, BasePolicy(), (2,0))

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    print(f"\033[32m{'─'*49}\n\t\tOptimizing Policy\n{'─'*49}\033[0m")
    policy = OptimalPolicy(
        maze=maze, 
        threshold=0.01,
        discount=1,
        probability=1,
        visualise=True
    )

    print(f"\033[32m{'─'*45}\n\t\tAgent actions\n{'─'*45}\033[0m")
    # Assign policy to agent
    agent.policy = policy
    print(agent)
    # keep going until terminate state is reached
    while not maze[agent.current_coordinate].is_terminal:
        agent.act(True)

def simulate_base_assignment_EXTRA()-> None:
    """
    Creates maze from assignment.
    Places agent in maze.
    Print maze with agent.
    Perform value iteration, given a probability.
    Extract optimal policy.
    Print both.
    Have agent perform this optimal policy in maze, 
    using the probability.
    """
    maze_shape = (4,4)
    probability = 0.7

    rewards = np.array([
        [10,  -1,  -1,  -1],
        [-2,  -1,  -1,  -1],
        [-1,  -1, -10,  -1],
        [-1,  -1, -10,  40],
    ], dtype=int) # reward matrix from assignment

    maze = StupidMaze(maze_shape, rewards)
    maze.set_terminal((0,0))
    maze.set_terminal((3,3))

    agent = ProbabilityAgent(maze, BasePolicy(), (2,0), probability)

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    print(f"\033[32m{'─'*49}\n\t\tOptimizing Policy\n{'─'*49}\033[0m")
    policy = OptimalPolicy(
        maze=maze, 
        threshold=0.01,
        discount=1,
        probability=probability,
        visualise=True
    )

    print(f"\033[32m{'─'*45}\n\t\tAgent actions\n{'─'*45}\033[0m")
    # Assign policy to agent
    agent.policy = policy

    print(agent)
    # keep going until terminate state is reached
    while not maze[agent.current_coordinate].is_terminal:
        agent.act(True)
