ROUND_OFF = 5
ACTIONS = ["goto", "approach", "opendoor", "gothrough"]

MAP = [""]

goal_nodes_tuple = (10,)
has_door_nodes_tuple = (1, 2, 4, 5, 6, 8, 9, 11, 12, 14,)
door_open_nodes_dict = {i: False for i in has_door_nodes_tuple}
door_id_dict = {1: 2, 14: 2,
                2: 0, 5: 0,
                4: 1, 6: 1,
                8: 4, 9: 4,
                11: 3, 12: 3}

success_rate_dict1 = {0: 1.0,
                      1: 1.0,
                      2: 1.0,
                      3: 1.0,
                      4: 1.0,
                      5: 1.0,
                      6: 1.0,
                      7: 1.0,
                      8: 1.0,
                      9: 1.0,
                      10: 1.0,
                      11: 1.0,
                      12: 1.0,
                      13: 1.0,
                      14: 1.0}

success_rate_dict2 = {0: 0.7,
                      1: 0.8,
                      2: 0.5,
                      3: 0.7,
                      4: 0.5,
                      5: 0.5,
                      6: 0.5,
                      7: 0.7,
                      8: 0.7,
                      9: 0.7,
                      10: 0.7,
                      11: 0.7,
                      12: 0.7,
                      13: 0.7,
                      14: 0.7}
