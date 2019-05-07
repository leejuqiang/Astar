import numpy as np
from PIL import Image

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observation(input_observation):
    #im0 = Image.fromarray(input_observation)
    processed_observation = remove_color(input_observation)
    processed_observation = processed_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 100 # everything else (paddles, ball) just set to 1
    # (80, 80)
    # subtract the previous frame from the current one so we are only processing on changes in the game
    # store the previous frame so we can subtract from it next time
    return np.reshape(processed_observation, (1, 80, 80))

def get_screen(env):
    observation = env.render(mode='rgb_array')
    return preprocess_observation(observation)
