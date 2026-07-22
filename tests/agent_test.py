learner = Learner(algo='sarsa', alpha=0.1, gamma=0.9)
epsilons = decay_schedule(1.0, 0.01, 0.9, n_episodes)

for e in range(n_episodes):
    learner.epsilon = epsilons[e]
    env.reset()
    state_dict = env.get_state(None)
    action, a_idx, s = learner.select_action(state_dict)

    done = False
    while not done:
        reward, done, score = env.play_step(action)
        next_state_dict = env.get_state(None)
        next_action, a2_idx, s2 = learner.select_action(next_state_dict)

        learner.train_step(s, a_idx, reward, s2, done, a2_idx)

        action, a_idx, s = next_action, a2_idx, s2

    learner.n_games += 1