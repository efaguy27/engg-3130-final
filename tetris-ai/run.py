from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import os
        

# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 2000
    max_steps = 1000000000
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 1
    log_every = 1
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = 0.01
    activations = ['relu', 'relu', 'linear']
    m = 0

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-eps={episodes}-e-stop={epsilon_stop_episode}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []
    steps_list = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0
        
        if (render_every and episode % render_every == 0) or episode == (episodes - 1):
            render = True
            record = True
        else:
            render = False
            record = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], episode, render=render,
                                    render_delay=render_delay, record=record)
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())
        steps_list.append(steps)

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = scores[-log_every]
            avg_steps = steps_list[-log_every]

            log.log(episode, avg_score=avg_score,
                   avg_steps=avg_steps)


if __name__ == "__main__":
    dqn()
