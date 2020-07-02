import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def policy(obs):
    player_sum, dealer_card, useable_ace = obs
    return 0 if player_sum >= 20 else 1

def generate_eps(policy, env):
    states = []
    rewards =[]
    obs = env.reset()
    done = False
    while not done:
        action = policy(obs)
        obs, reward, done, _ = env.step(action)
        states.append(obs)
        rewards.append(reward)

    return states, rewards

def frist_visit_mc_predict(policy, env, eps):
    V, returns = {}, {}
    for i in range(eps):
        G = 0
        states, rewards = generate_eps(policy, env)
        visited_state = set()

        for t in range(len(states)-1, -1, -1):
            s = states[t]
            r = rewards[t]
            G += r

            if s not in visited_state:
                reward, visits = returns.get(s, [0, 0])
                returns[s] = [reward + G, visits + 1]
                visited_state.add(s)

    for s , (sum_reward, n_visits) in returns.items():
        V[s] = sum_reward/n_visits

    return V

def plot_fun(V, ax1, ax2):
    player_sum = np.arange(12, 22)
    dealer_card = np.arange(1, 11)
    usable_ace = np.array([True, False])
    s_v = np.zeros((len(player_sum), len(dealer_card), len(usable_ace)))

    for i, p in enumerate(player_sum):
        for j, d in enumerate(dealer_card):
            for k, ua in enumerate(usable_ace):
                s_v[i, j, k] = V.get((p, d, ua), 0)

    x, y = np.meshgrid(dealer_card, player_sum)

    #without useable ace
    ax1.plot_wireframe(x, y, s_v[:, :, 0])
    ax1.set_zlim(-1,1)
    ax1.set_xlabel('Dealer showing')
    ax1.set_ylabel('Player sum')
    #ax1.set_zlabel('value')

    #with useable ace
    ax2.plot_wireframe(x, y, s_v[:, :, 1])
    ax2.set_zlim(-1, 1)
    ax2.set_xlabel('Dealer showing')
    ax2.set_ylabel('Player sum')
    #ax2.set_zlabel('value')


def generate_eps_from_policy(env, pi):
    states = []
    rewards = []
    actions = []
    state = env.reset()
    done = False
    while not done:
        # select the best action according to the policy pi if it exists
        # otherwise choose 0 (arbitrary policy not changed so far)
        action = pi[state] if state in pi else 0
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards


def mc_es(env, generate_eps_from_policy, episodes):
    n_actions = env.action_space.n

    # initialize
    pi = {}
    Q = defaultdict(lambda: np.zeros(n_actions))
    returns = {}

    for episode in range(episodes):
        states, actions, rewards = generate_eps_from_policy(env, pi)
        G= 0

        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            G += reward

            visited_state = set()
            if (state, action) not in visited_state:
                # append G to returns (s_t, a_t)
                reward, visits = returns.get((state, action), [0, 0])
                returns[(state, action)] = [reward + G, visits + 1]

                Q[state][action] = (reward + G) / (visits + 1)
                # update policy greedily
                pi[state] = np.argmax(Q[state])
                visited_state.add((state, action))
    #get value function
    v = {state: np.max(values) for state, values in Q.items()}

    return v, pi

def plot_value(v, eps):
    #v, pi = mc_es(env, generate_eps_from_policy, 0.5, 10000)
    #print(v, '\n')
    #print(pi, '\n')
    # generate_eps(policy, env)
    fig, axes = plt.subplots(nrows=2, subplot_kw={'projection': '3d'})
    fig.suptitle("After %d episodes"%eps, fontsize=16)
    axes[0].set_title('Uable ace')
    axes[1].set_title('No usable ace')
    plot_fun(v, axes[0], axes[1])
    plt.show()

def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')

    eps1 = 10000
    eps2 = 500000

    v1 = frist_visit_mc_predict(policy, env, eps1)
    plot_value(v1, eps1)

    #v2 = frist_visit_mc_predict(policy, env, eps2)
    #plot_value(v2, eps2)

    v3, pi3 = mc_es(env, generate_eps_from_policy, eps1)
    #plot_value(v3, eps1)
    #print(len(pi3))


    v4, pi4 = mc_es(env, generate_eps_from_policy, eps2)
    plot_value(v4, eps2)
    print(len(pi4))


    '''
    eps = 10000 #10000
    v = frist_visit_mc_predict1(policy, env, eps)
    fig, axes = plt.subplots(nrows =  2, subplot_kw={'projection': '3d'})
    fig.suptitle("After 10,000 episodes", fontsize=16)
    axes[0].set_title('Uable ace')
    axes[1].set_title('No usable ace')
    plot_value(v, axes[0], axes[1])
    plt.show()

    eps1 = 500000
    v1 = frist_visit_mc_predict(policy, env, eps1)
    fig, axes = plt.subplots(nrows=2, subplot_kw={'projection': '3d'})
    fig.suptitle("After 500,000 episodes", fontsize=16)
    axes[0].set_title('No usable ace')
    axes[1].set_title('Usable ace')
    plot_value(v1, axes[0], axes[1])
    plt.show()

    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False

    while not done:
        #print('test policy', policy(obs))
        print("observation:", obs)
        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)
        print("reward:", reward)
        print("")

    '''


if __name__ == "__main__":
    main()