import numpy as np

ROUND_OFF = 5
ACTIONS = ["goto", "approach", "opendoor", "gothrough"]


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


node_num = 27
goal_nodes = (14, 16)
start_nodes = (0, 5, 21)

has_door_nodes_tuple = (1, 2, 3, 4, 8, 9, 11, 12, 16, 17, 19, 20)
door_open_nodes_dict = {i: False for i in has_door_nodes_tuple}
goal_reward = 1000
stack_cost = 1000

door_id_dict = {1: 0,
                2: 1,
                3: 0,
                4: 1,
                8: 2,
                9: 2,
                11: 3,
                12: 3,
                16: 4,
                17: 4,
                19: 5,
                20: 5}

success_rate_dict0 = {0: [1.0, 0.0, 0.0],
                      1: [1.0, 0.0, 0.0],
                      2: [1.0, 0.0, 0.0],
                      3: [1.0, 0.0, 0.0],
                      4: [1.0, 0.0, 0.0],
                      5: [1.0, 0.0, 0.0],
                      6: [1.0, 0.0, 0.0],
                      7: [1.0, 0.0, 0.0],
                      8: [1.0, 0.0, 0.0],
                      9: [1.0, 0.0, 0.0],
                      10: [1.0, 0.0, 0.0],
                      11: [1.0, 0.0, 0.0],
                      12: [1.0, 0.0, 0.0],
                      13: [1.0, 0.0, 0.0],
                      14: [1.0, 0.0, 0.0],
                      15: [1.0, 0.0, 0.0],
                      16: [1.0, 0.0, 0.0],
                      17: [1.0, 0.0, 0.0],
                      18: [1.0, 0.0, 0.0],
                      19: [1.0, 0.0, 0.0],
                      20: [1.0, 0.0, 0.0],
                      21: [1.0, 0.0, 0.0],
                      22: [1.0, 0.0, 0.0],
                      23: [1.0, 0.0, 0.0],
                      24: [1.0, 0.0, 0.0],
                      26: [1.0, 0.0, 0.0],
                      25: [1.0, 0.0, 0.0]}

success_rate_dict1 = {0: [0.99, 0.01, 0.0],
                      1: [0.7, 0.1, 0.2],
                      2: [0.99, 0.0, 0.01],
                      3: [0.7, 0.1, 0.2],
                      4: [0.99, 0.0, 0.01],
                      5: [0.99, 0.01, 0.0],
                      6: [0.99, 0.01, 0.0],
                      7: [0.99, 0.01, 0.0],
                      8: [0.7, 0.1, 0.2],
                      9: [0.7, 0.1, 0.2],
                      10: [0.99, 0.01, 0.0],
                      11: [0.99, 0.0, 0.01],
                      12: [0.99, 0.0, 0.01],
                      13: [0.99, 0.01, 0.0],
                      14: [0.99, 0.01, 0.0],
                      15: [0.99, 0.01, 0.0],
                      16: [0.85, 0.1, 0.05],
                      17: [0.85, 0.1, 0.05],
                      18: [0.99, 0.01, 0.0],
                      19: [0.8, 0.1, 0.1],
                      20: [0.8, 0.1, 0.1],
                      21: [0.99, 0.01, 0.0],
                      22: [0.99, 0.01, 0.0],
                      23: [0.99, 0.01, 0.0],
                      24: [0.99, 0.01, 0.0],
                      25: [0.99, 0.01, 0.0],
                      26: [0.99, 0.01, 0.0]}

success_rate_dict2 = {0: [0.9, 0.1, 0.0],
                      1: [0.89, 0.1, 0.01],
                      2: [0.89, 0.1, 0.01],
                      3: [0.89, 0.1, 0.01],
                      4: [0.89, 0.1, 0.01],
                      5: [0.9, 0.1, 0.0],
                      6: [0.9, 0.1, 0.0],
                      7: [0.9, 0.1, 0.0],
                      8: [0.85, 0.1, 0.05],
                      9: [0.85, 0.1, 0.05],
                      10: [0.9, 0.1, 0.0],
                      11: [0.89, 0.1, 0.01],
                      12: [0.89, 0.1, 0.01],
                      13: [0.9, 0.1, 0.0],
                      14: [0.9, 0.1, 0.0],
                      15: [0.9, 0.1, 0.0],
                      16: [0.9, 0.1, 0.0],
                      17: [0.9, 0.1, 0.0],
                      18: [0.9, 0.1, 0.0],
                      19: [0.89, 0.1, 0.01],
                      20: [0.89, 0.1, 0.01],
                      21: [0.9, 0.1, 0.0],
                      22: [0.9, 0.1, 0.0],
                      23: [0.9, 0.1, 0.0],
                      24: [0.9, 0.1, 0.0],
                      25: [0.9, 0.1, 0.0],
                      26: [0.9, 0.1, 0.0]}

success_rate_dict3 = {0: [1.0, 0.0, 0.0],
                      1: [0.8, 0.01, 0.19],
                      2: [0.75, 0.1, 0.15],
                      3: [0.8, 0.01, 0.19],
                      4: [0.75, 0.1, 0.15],
                      5: [1.0, 0.0, 0.0],
                      6: [1.0, 0.0, 0.0],
                      7: [1.0, 0.0, 0.0],
                      8: [0.65, 0.05, 0.3],
                      9: [0.65, 0.05, 0.3],
                      10: [1.0, 0.0, 0.0],
                      11: [0.6, 0.01, 0.39],
                      12: [0.6, 0.01, 0.39],
                      13: [1.0, 0.0, 0.0],
                      14: [1.0, 0.0, 0.0],
                      15: [1.0, 0.0, 0.0],
                      16: [0.5, 0.1, 0.4],
                      17: [0.5, 0.1, 0.4],
                      18: [1.0, 0.0, 0.0],
                      19: [0.9, 0.01, 0.09],
                      20: [0.9, 0.01, 0.09],
                      21: [1.0, 0.0, 0.0],
                      22: [1.0, 0.0, 0.0],
                      23: [1.0, 0.0, 0.0],
                      24: [1.0, 0.0, 0.0],
                      25: [1.0, 0.0, 0.0],
                      26: [1.0, 0.0, 0.0]}

success_rate_dict4 = {0: [0.9, 0.1, 0.0],
                      1: [0.62, 0.19, 0.19],
                      2: [0.7, 0.15, 0.15],
                      3: [0.62, 0.19, 0.19],
                      4: [0.7, 0.15, 0.15],
                      5: [0.9, 0.1, 0.0],
                      6: [0.9, 0.1, 0.0],
                      7: [0.8, 0.2, 0.0],
                      8: [0.55, 0.15, 0.3],
                      9: [0.55, 0.15, 0.3],
                      10: [0.9, 0.1, 0.0],
                      11: [0.5, 0.11, 0.39],
                      12: [0.5, 0.11, 0.39],
                      13: [0.8, 0.2, 0.0],
                      14: [0.8, 0.2, 0.0],
                      15: [0.8, 0.2, 0.0],
                      16: [0.5, 0.2, 0.3],
                      17: [0.5, 0.2, 0.3],
                      18: [0.8, 0.2, 0.0],
                      19: [0.8, 0.11, 0.09],
                      20: [0.8, 0.11, 0.09],
                      21: [0.9, 0.1, 0.0],
                      22: [0.9, 0.1, 0.0],
                      23: [0.9, 0.1, 0.0],
                      24: [0.9, 0.1, 0.0],
                      25: [0.9, 0.1, 0.0],
                      26: [0.9, 0.1, 0.0]}
