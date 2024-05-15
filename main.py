import baseAssignmentSimulations as bas
import baseAssignmentSimulationsGUI as gui
import randomExtraAssignmentSimulations as extra



def main()-> None:

    """ Base assignment, using CLI interface """
    bas.simulate_base_assignment_A()
    # bas.simulate_base_assignment_B()
    # bas.simulate_base_assignment_C()
    # bas.simulate_base_assignment_EXTRA() # 70%


    """ Base assignment, using GUI interface """
    # gui.simulate_base_assignment_A()
    # gui.simulate_base_assignment_B()
    # gui.simulate_base_assignment_C()
    # gui.simulate_base_assignment_EXTRA() # 70%


    """ Extra simulations that showcase the codes extra capabilities """
    # NOTE: None of these are stupid mazes

    # extra.simulate_probability_5x10_grid_w_random_reward(
    #     let_agent_play=False
    # )
    # extra.simulate_probability_2x2_grid_w_random_reward_GUI(
    #     let_agent_play=True
    # )

if __name__ == "__main__":
    main()
