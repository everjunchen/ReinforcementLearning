import gym
import numpy as np
import matplotlib.pyplot as plt

def value_iteration(env):
    # Init some useful variables:
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    count = 0
    policy = np.zeros(n_states)

    for i in range(1000000):
        # count the iteration
        count += 1
        delta = 0
        for s in range(n_states):
            v = V_states[s]
            a_value = np.zeros(n_actions)
            for a in range(n_actions):
                q = 0
                for p, n_state, r, is_terminal in env.P[s][a]:
                    q += p*(r + gamma*V_states[n_state])
                a_value[a] = q
            #update V_states
            V_states[s] = np.max(a_value)
            policy[s] = np.argmax(a_value)
            delta = max(delta, abs(v - V_states[s]))
            #print(count, V_states)
        if (delta < theta):
            print('steps to converge:', count)
            print('optimal value function:\n', V_states)
            break
    return V_states

def epsilon_greedy_action(env, epsilon, Q, state):
    if np.random.uniform(0, 1) < epsilon:
        a = np.random.randint(env.action_space.n)  #
    else:
        a = np.argmax(Q[state, :])
    return a

def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    scores = []

    for i in range(num_ep):
        state_m = {}
        action_m = {}
        reward_m = {}
        done = False
        score = 0
        tau = 0
        t = -1
        T = np.inf

        s = env.reset()
        a = epsilon_greedy_action(env, epsilon, Q, s)
        action_m[0] = a
        state_m[0] = s

        while tau < (T - 1):
            t += 1
            if t < T:
                s_, r, done, _ = env.step(a)
                score += r
                state_m[(t + 1) % n] = s_
                reward_m[(t + 1) % n] = r
                if done:
                    T = t + 1
                    print('eps ends at step', t)
                else:
                    a = epsilon_greedy_action(env, epsilon, Q, s)
                    action_m[(t + 1) % n] = a

            tau = t - n + 1

            if tau >= 0:
                G = np.sum([gamma ** (j - tau - 1) * reward_m[j % n] for j in range(tau + 1, min(tau + n, T) + 1)])

                if tau + n < T:
                    G += gamma ** n * Q[int(state_m[(tau + n) % n])][int(action_m[(tau + n) % n])]

                s_tn = int(state_m[(tau) % n])
                a_tn = int(action_m[(tau) % n])

                Q[s_tn][a_tn] += alpha * (G - Q[s_tn][a_tn])
            #print('tau', 'Q', Q)
            #Q[state_m[tau % n]][action_m[tau % n]]
    print(Q)
    return Q

env=gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
true_V = value_iteration(env)
print(true_V)
for i in np.arange(0, 1.0, 0.1):
    #print('alpha',i)
    n = 2 ** np.arange(9)
    for j in n:
        print('alpha', i, 'n:', j)
        nstep_sarsa(env, n=j, alpha=i)
#nstep_sarsa(env)