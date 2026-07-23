import numpy as np
import random

#action - (straight , left , right)

class Learner:
    def __init__(self , algo, n_episodes , n_states=2048, n_actions=3, alpha=0.1, gamma=0.9, lambda_=0.5 , n_planning = 20 , trajectory_depth = 20, replacing_traces=True):
        self.n_games = 0
        self.algo = algo
        self.epsilon = 1.0
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.replacing_traces = replacing_traces
        self.n_states = n_states
        self.n_episodes = n_episodes
        self.n_planning = n_planning
        self.trajectory_depth = trajectory_depth

        self.decay_values = self._decay_schedule(1.0, 0.05, 0.5, self.n_episodes) # will need to finetune these knobs later

        self.Q = np.zeros((n_states , n_actions))
        if self.algo == "double_q":
            self.Q2 = np.zeros((n_states , n_actions))
        if self.algo == "sarsa_lambda" or self.algo == "q_lambda":
            self.E = np.zeros((n_states , n_actions))
        if self.algo == "dyna_q" or self.algo == "trajectory_sampling":
            self.T_count = np.zeros((n_states , n_actions , n_states))
            self.R_model = np.zeros((n_states , n_actions , n_states))
            self.Done_count = np.zeros((n_states, n_actions, n_states))

        self.episode_buffer = [] # for MC


    def _to_index(self, state_dict):
        idx = 0
        for bit in state_dict.values():
            idx = (idx << 1) | int(bit)
        return idx

    def _decay_schedule(self , init_value , min_value , decay_ratio , max_steps):
        decay_steps = (int) (decay_ratio * max_steps)
        rem_steps = max_steps - decay_steps

        values = np.logspace(-2 , 0 , decay_steps , base=10 , endpoint=True)[::-1] # reverse to get - inverse log decay
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values , (0 , rem_steps) , 'edge')
        return values

    def select_action(self, state_dict , epsilon) -> tuple:
        state = self._to_index(state_dict)
        action = [0, 0, 0]

        if np.random.random() < epsilon:
            move = random.randint(0, 2)
        else:
            Q_eff = self.Q if self.algo != 'double_q' else (self.Q + self.Q2) / 2
            move = np.argmax(Q_eff[state])

        action[move] = 1
        return action , move , state

    def train_step(self , state, action, reward, next_state, done, next_action=None):
        if self.algo == "mc":
            self.episode_buffer.append((state , action , reward))
            if done:
                self._mc_flush()
            return

        if self.algo == "sarsa":
            target = reward + self.gamma * self.Q[next_state][next_action] * (not done)
            self._apply(self.Q , state , action , target)

        elif self.algo == "q":
            target = reward + self.gamma * self.Q[next_state].max() * (not done)
            self._apply(self.Q , state , action , target)

        elif self.algo == "double_q":
            if np.random.randint(2):
                action_star = np.argmax(self.Q[next_state])
                target = reward + self.gamma * self.Q2[next_state][action_star] * (not done) # target from alternate Q func
                self._apply(self.Q, state, action, target) # update gets applied to the original Q func
            else:
                action_star = np.argmax(self.Q2[next_state])
                target = reward + self.gamma * self.Q[next_state][action_star] * (not done) # target from alternate Q func
                self._apply(self.Q2, state, action, target) # update gets applied to the original Q func

        elif self.algo == "sarsa_lambda":
            target = reward + self.gamma * self.Q[next_state][next_action] * (not done)
            td_error = target - self.Q[state][action]

            self.E[state][action] += 1
            if self.replacing_traces: np.clip(self.E , 0 , 1 , out = self.E)
            self.Q += self.alpha * td_error * self.E
            self.E *= self.lambda_ * self.gamma

            if done: self.E.fill(0)

        elif self.algo == "q_lambda":
            next_action_greedy = self.Q[next_state].max() == self.Q[next_state][next_action]
            target = reward + self.gamma * self.Q[next_state].max() * (not done)
            td_error = target - self.Q[state][action]

            self.E[state][action] += 1
            if self.replacing_traces: np.clip(self.E , 0 , 1 , out = self.E)
            self.Q += self.alpha * td_error * self.E
            self.E *= self.lambda_ * self.gamma if next_action_greedy else 0

            if done: self.E.fill(0)

        elif self.algo == "dyna_q" or self.algo == "trajectory_sampling":
            self.T_count[state][action][next_state] += 1
            self.Done_count[state][action][next_state] += int(done)
            r_diff = reward - self.R_model[state][action][next_state]
            self.R_model[state][action][next_state] += r_diff / self.T_count[state][action][next_state]

            td_target = reward + self.gamma * self.Q[next_state].max() * (not done)
            self._apply(self.Q , state , action , td_target)

            if self.algo == "dyna_q":
                if self.algo == "dyna_q":
                    for _ in range(self.n_planning):
                        if self.Q.sum() == 0: break

                        visited_states = np.where(np.sum(self.T_count, axis=(1, 2)) > 0)[0]
                        if len(visited_states) == 0: break
                        sim_state = np.random.choice(visited_states)

                        actions_taken = np.where(np.sum(self.T_count[sim_state], axis=1) > 0)[0]
                        sim_action = np.random.choice(actions_taken)

                        probs = self.T_count[sim_state][sim_action] / self.T_count[sim_state][sim_action].sum()
                        sim_next_state = np.random.choice(np.arange(self.n_states), p=probs)

                        sim_reward = self.R_model[sim_state][sim_action][sim_next_state]

                        # mask terminal states
                        sim_done_prob = self.Done_count[sim_state][sim_action][sim_next_state] / self.T_count[sim_state][sim_action][sim_next_state] # get probability of terminal state
                        sim_done = (np.random.random() < sim_done_prob)

                        sim_td_target = sim_reward + self.gamma * self.Q[sim_next_state].max() * (not sim_done)
                        self._apply(self.Q, sim_state, sim_action, sim_td_target)

            else:
                curr_sim_state = state
                for _ in range(self.trajectory_depth):
                    if self.Q.sum() == 0:
                        break
                    sim_action = self.Q[curr_sim_state].argmax()

                    count_sum = self.T_count[curr_sim_state][sim_action].sum()
                    if count_sum == 0:
                        break  # Unvisited state-action pair

                    probs = self.T_count[curr_sim_state][sim_action] / count_sum
                    sim_next_state = np.random.choice(np.arange(self.n_states), p=probs)

                    sim_reward = self.R_model[curr_sim_state][sim_action][sim_next_state]

                    #mask terminal states
                    sim_done_prob = self.Done_count[curr_sim_state][sim_action][sim_next_state] / self.T_count[curr_sim_state][sim_action][sim_next_state]
                    sim_done = (np.random.random() < sim_done_prob)

                    sim_td_target = sim_reward + self.gamma * self.Q[sim_next_state].max() * (not sim_done)
                    self._apply(self.Q, curr_sim_state, sim_action, sim_td_target)

                    if sim_done: break
                    curr_sim_state = sim_next_state

    def _apply(self , table , s , a , target):
        td_error = target - table[s][a]
        table[s][a] += self.alpha * td_error

    def _mc_flush(self):
        G = 0
        visited = set()
        for s , a , r in reversed(self.episode_buffer):
            G = self.gamma * G + r # bellman backup
            if (s , a) in visited:
                continue
            visited.add((s , a))
            self._apply(self.Q , s , a , G)

        self.episode_buffer.clear()