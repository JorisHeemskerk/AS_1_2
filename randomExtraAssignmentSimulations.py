import numpy as np
import pygame

from basePolicy import BasePolicy
from optimalPolicy import OptimalPolicy
from optimalPolicyGUI import OptimalPolicyGUI
from probabilityAgent import ProbabilityAgent
from stupidMaze import BaseMaze
from utils import draw_matrix, put_agent_colour_in_colour_matrix, BLACK, WINDOW_SIZE


def simulate_probability_5x10_grid_w_random_reward(
        let_agent_play: bool
    )-> None:
    """
    Creates 5x10 maze with terminal states 4,4 and 1,8.
    Creates probability agent with probability of 0.8
    Creates optimal policy with discount of 0.9

    @param let_agent_play: let agent perform actions if true, else not
    """
    maze_shape = (5,10)
    probability = 0.8

    rewards = np.random.randint(-10, 10, size=maze_shape)

    maze = BaseMaze(maze_shape, rewards)
    maze.set_terminal((4,4))
    maze.set_terminal((1,8))

    agent = ProbabilityAgent(maze, BasePolicy(), (1,1), probability)

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    print(f"\033[32m{'─'*49}\n\t\tOptimizing Policy\n{'─'*49}\033[0m")
    policy = OptimalPolicy(
        maze=maze, 
        threshold=0.01,
        discount=.9,
        probability=probability,
        visualise=True
    )

    print(f"\033[32m{'─'*45}\n\t\tAgent actions\n{'─'*45}\033[0m")
    # Assign policy to agent
    agent.policy = policy

    print(agent)

    if let_agent_play:
        # keep going until terminate state is reached
        while not maze[agent.current_coordinate].is_terminal:
            agent.act(True)

def simulate_probability_2x2_grid_w_random_reward_GUI(
        let_agent_play: bool
    )-> None:
    """
    Creates 5x10 maze with terminal state 2,2.
    Creates probability agent with probability of 0.8
    Creates optimal policy with discount of 0.9

    @param let_agent_play: let agent perform actions if true, else not
    """
    maze_shape = (2,2)
    probability = 0.8

    rewards = np.random.randint(0, 10, size=maze_shape)

    maze = BaseMaze(maze_shape, rewards)
    maze.set_terminal((1,1))

    agent = ProbabilityAgent(maze, BasePolicy(), (0,0), probability)

    colour_matrix = np.array([
        [BLACK if not state.is_terminal else (255, 0, 0) for state in row] 
        for row in maze.states
    ]).T[:, ::-1]

    policy = OptimalPolicyGUI(
        maze=maze, 
        threshold=0.01,
        discount=.9,
        colour_matrix=colour_matrix,
        probability=probability
    )

    if let_agent_play:
        # Assign policy to agent
        agent.policy = policy

        pygame.init()
        font = pygame.font.SysFont(None, 20)
        screen = pygame.display.set_mode(WINDOW_SIZE)
        data_matrix = np.array(
            [[f"r = {state.reward}" for state in row] for row in maze.states]
        ).T[::-1]
        
        pygame.display.set_caption("ProbabilityAgent with OptimalPolicyGUI")

        draw_matrix(
            data_matrix, 
            put_agent_colour_in_colour_matrix(
                colour_matrix.copy(),
                agent.current_coordinate,
                (155,155,0)
            ), 
            screen, 
            font
        )
        
        running = True
        while running:
            pygame.time.delay(500)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not maze[agent.current_coordinate].is_terminal:
                agent.act()
            draw_matrix(
                data_matrix, 
                put_agent_colour_in_colour_matrix(
                    colour_matrix.copy(),
                    agent.current_coordinate,
                    (155,155,0)
                ), 
                screen, 
                font
            )
        pygame.quit()
