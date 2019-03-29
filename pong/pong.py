"""
source: https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0
"""
import gym
import numpy as np

UP = 2
DOWN = 3

def downsample(image):
    # Take every other pixel
    return image[::2, ::2, :]

def remove_color(image):
    # Convert the colors
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    return image

"""
    1. crop the image (get only relevant info)
    2. downsample the image
    3. Convert to black and white
    4. remove background
    5. convert 80 x 80 to 6400 x 1
    6. Store difference between current and previous frame
"""
def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    processed_observation = input_observation[35:195]
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # set others (paddles, ball) to 1

    # Convert 80x80 to 1600x1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtrace previous frame so that we only get the changes
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations

def relu(vector):
    vector[vector < 0] = 0
    return vector

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def apply_neural_nets(observation_matrix, weights):
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        return UP
    else:
        return DOWN

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
            '1': dC_dw1,
            '2': dC_dw2
            }


def main():
    env = gym.make("Pong-v0")
    observation = env.reset()

    # set the params
    batch_size = 10 # rounds to play before updating weights
    gamma = 0.99 # discount factor
    decay_rate = 0.99
    num_hidden_layer_neurons = 200 # neurons in hidden layer
    input_dimensions = 80 * 80 # dimension of observation image
    learning_rate = 1e-4

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
        }

    # rmsprop algorithm
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards =\
            [], [], [], []

    while True:
        env.render()
        processed_observations, prev_processed_observations =\
                preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values = up_probability = apply_neural_nets(processed_observations, weights)

        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # do the action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)




if __name__ == "__main__":
    main()
