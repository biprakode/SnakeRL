import numpy as np

from game import SnakeGame
from learner import Learner


def train(algo , n_episodes , save_path, **kwargs):
    env = SnakeGame(w = 320 , h = 240)
    learner = Learner(algo , n_episodes , **kwargs)
    scores = np.zeros(n_episodes)

    for e in range(n_episodes):
        epsilons = learner.decay_values[e]
        env.reset()
        state_dict = env.get_state()
        action , a_idx , state = learner.select_action(state_dict , epsilons)
        done = False
        score = -1

        while not done:
            reward , done , score = env.play_step(action)
            next_state_dict = env.get_state()
            next_action , a2_idx , next_state = learner.select_action(next_state_dict , epsilons)

            learner.train_step(state_dict , a_idx , reward , next_state, done , a2_idx)

            action , a_idx , state = next_action , a2_idx , next_state

        learner.n_games +=1
        scores[e] = score

        if e % 100 == 0:
            print(f"[{algo}] episode {e}/{n_episodes}  score={score}  "
                  f"mean_last_100={scores[max(0,e-100):e+1].mean():.2f}")

    np.save(save_path, scores)
    return scores, learner
