import numpy as np

NUM_DIGITIZED = 6


def bin(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num)[1:-1]


def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitize = [
        np.digitize(cart_pos, bins=bin(-2.4, 2.4, NUM_DIGITIZED)),
        np.digitize(cart_v, bins=bin(-3., 3., NUM_DIGITIZED)),
        np.digitize(pole_angle, bins=bin(-0.5, 0.5, NUM_DIGITIZED)),
        np.digitize(pole_v, bins=bin(-2.0, 2.0, NUM_DIGITIZED))
    ]
    return sum([x * (NUM_DIGITIZED ** i) for i, x in enumerate(digitize)])
