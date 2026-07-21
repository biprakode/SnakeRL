import numpy as np
from game import SnakeGame, Direction

def random_action():
    a = [0, 0, 0]
    a[np.random.randint(3)] = 1
    return a

def test_basic_loop(n_episodes=20):
    env = SnakeGame()
    for ep in range(n_episodes):
        env.reset()
        steps = 0
        done = False
        while not done:
            action = random_action()
            reward, done, score = env.play_step(action)
            steps += 1
            assert reward in (-10, 0, 10), f"unexpected reward: {reward}"
        print(f"episode {ep}: score={score}, steps={steps}, "
              f"terminated_by_timeout={steps >= 100*len(env.snake)}")

def test_state_vector():
    env = SnakeGame()
    env.reset()
    state = env.get_state(None)
    print("state dict:", state)
    print("all keys boolean?", all(isinstance(v, bool) for v in state.values()))
    idx = env.state_to_index(list(state.values()))
    print("packed index:", idx, "in range [0,2048)?", 0 <= idx < 2048)

if __name__ == "__main__":
    test_basic_loop()
    test_state_vector()