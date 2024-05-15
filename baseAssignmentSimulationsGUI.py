import numpy as np
import pygame

from baseAgent import BaseAgent
from basePolicy import BasePolicy
from optimalPolicyGUI import OptimalPolicyGUI
from probabilityAgent import ProbabilityAgent
from stupidMaze import StupidMaze
from utils import draw_matrix, put_agent_colour_in_colour_matrix, BLACK, WINDOW_SIZE


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

    pygame.init()
    font = pygame.font.SysFont(None, 20)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    data_matrix = np.array(
        [[f"r = {state.reward}" for state in row] for row in maze.states]
    ).T[::-1]
    colour_matrix = np.array([
        [(255, 0, 0), BLACK, BLACK, BLACK],
        [BLACK, BLACK, BLACK, BLACK],
        [BLACK, BLACK, (0, 0, 255), BLACK],
        [BLACK, BLACK, (0, 0, 255), (255, 0, 0)],
    ]).T[:, ::-1]
    pygame.display.set_caption("Maze layout")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            draw_matrix(data_matrix, colour_matrix, screen, font)
        pygame.time.delay(300)
    pygame.quit()

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

    pygame.init()
    font = pygame.font.SysFont(None, 20)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    data_matrix = np.array(
        [[f"r = {state.reward}" for state in row] for row in maze.states]
    ).T[::-1]
    colour_matrix = np.array([
        [(255, 0, 0), BLACK, BLACK, BLACK],
        [BLACK, BLACK, BLACK, BLACK],
        [BLACK, BLACK, (0, 0, 255), BLACK],
        [BLACK, BLACK, (0, 0, 255), (255, 0, 0)],
    ]).T[:, ::-1]
    pygame.display.set_caption("BaseAgent with BasePolicy")

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

    colour_matrix = np.array([
            [(255, 0, 0), BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, (0, 0, 255), BLACK],
            [BLACK, BLACK, (0, 0, 255), (255, 0, 0)],
        ]).T[:, ::-1]

    policy = OptimalPolicyGUI(
        maze=maze, 
        threshold=0.01,
        discount=1,
        colour_matrix=colour_matrix,
        probability=1
    )

    # Assign policy to agent
    agent.policy = policy

    pygame.init()
    font = pygame.font.SysFont(None, 20)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    data_matrix = np.array(
        [[f"r = {state.reward}" for state in row] for row in maze.states]
    ).T[::-1]
    pygame.display.set_caption("BaseAgent with OptimalPolicyGUI")

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
        pygame.time.delay(500)
    pygame.quit()

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

    colour_matrix = np.array([
        [(255, 0, 0), BLACK, BLACK, BLACK],
        [BLACK, BLACK, BLACK, BLACK],
        [BLACK, BLACK, (0, 0, 255), BLACK],
        [BLACK, BLACK, (0, 0, 255), (255, 0, 0)],
    ]).T[:, ::-1]

    policy = OptimalPolicyGUI(
        maze=maze, 
        threshold=0.01,
        discount=1,
        colour_matrix=colour_matrix,
        probability=probability
    )

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
