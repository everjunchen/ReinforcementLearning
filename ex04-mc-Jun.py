import gym
import numpy as np
from collections import defaultdict

from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from functools import partial
#%matplotlib inline
#plt.style.use('ggplot')

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

def policy(obs):
    player_sum, dealer_card, useable_ace = obs
    return 0 if player_sum >= 20 else 1

def generate_eps(policy, env):
    states = []
    rewards =[]
    obs = env.reset()
    done = False
    while not done:
        #print('obs',obs)
        action = policy(obs)
        obs, reward, done, _ = env.step(action)
        #print('policy, reward',action, reward)
        states.append(obs)
        rewards.append(reward)

    return states, rewards

def frist_visit_mc_predict(policy, env, eps):
    V = defaultdict(float)
    counter = defaultdict(int)

    for i in range(eps):
        returns = 0
        states, rewards = generate_eps(policy, env)

        for t in range(len(states)-1, -1, -1):
            s = states[t]
            r = rewards[t]
            returns += r

            if s not in states[:t]:
                counter[s] += 1
                V[s] += (returns - V[s])/counter[s]
    print(V)
    return V

def frist_visit_mc_predict1(policy, env, eps):
    #V, returns = {}, {}
    V  = defaultdict(float)
    returns = {}
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
    print(V)

    return V

def plot_value(V, ax1, ax2):
    V = defaultdict(int, V)
    player_sum = np.arange(12, 22)
    dealer_card = np.arange(1, 11)
    usable_ace = np.array([True, False])
    s_v = np.zeros((len(player_sum), len(dealer_card), len(usable_ace)))

    for i, p in enumerate(player_sum):
        for j, d in enumerate(dealer_card):
            for k, ua in enumerate(usable_ace):
                s_v[i, j, k] = V[p, d, ua]
                #s = (p, d, ua)
                #s_v[i, j, k] = V.get(s, 0)

    x, y = np.meshgrid(dealer_card, player_sum)
    #print(x, y)

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

def plot_state_values(V, iterations):
	x_range = np.arange(12, 22) # if < 11 we are not dumb and we pick another card
	y_range = np.arange(1, 11) # dealer's face up card
	X, Y = np.meshgrid(x_range, y_range)

	fig = plt.figure(figsize=(15,20))

	aces = [(True, "Usable ace"), (False, "No usable ace")]

	for i, (usableAce, title) in enumerate(aces, 1):
		ax = fig.add_subplot(int("21" + str(i)), projection='3d')
		Z = np.array([V[x, y, usableAce] if (x, y, usableAce) in V else 0 for y in Y[:,0] for x in X[0] ]).reshape(X.shape)

		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, vmin=-1.0, vmax=1.0,
							   edgecolor='w', linewidth=0.5)
		ax.set_xlabel("Player's Current Sum")
		ax.set_ylabel("Dealer's Showing Card")
		ax.set_zlabel("Value")
		ax.view_init(ax.elev, -120)
		ax.set_title(title, fontsize=14)

	fig.suptitle("Value after %i iterations" % iterations, fontsize=20)
	plt.show()

def mc_es():
    player_sum = np.arange(12, 22)
    dealer_card = np.arange(1, 11)
    usable_ace = np.array([True, False])
    action = np.array(([0,1]))
    states = np.zeros((len(player_sum), len(dealer_card), len(usable_ace)))
    print(len(action), states.shape[1])

def generate_policy():
    action = np.random.choice(np.arange(2))
    return action

def generate_eps_from_policy(env, pi):
    states = []
    rewards = []
    actions = []
    state = env.reset()
    episode = []
    done = False
    while not done:
        # select the best action according to the policy pi if it exists
        # otherwise choose 0 (arbitrary policy not changed so far)
        action = pi[state] if state in pi else 0
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        episode.append((state, action, reward))
        state = next_state
    return states, actions, rewards
    #return episode


def mc_es(env, generate_eps_from_policy, gamma, episodes):
    n_actions = env.action_space.n

    # initialize arbitrarily pi, Q, returns
    pi = {}
    #pi = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(n_actions))
    returns = {}
    #returns = defaultdict(float)

    for episode in range(episodes):
        states, actions, rewards = generate_eps_from_policy(env, pi)
        #trajectory = generate_eps_from_policy(env, pi)
        G= 0

        for i in range(len(states) - 1, -1, -1):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            #G += gamma ** i * reward
            G += reward

            visited_state = set()
            if (state, action) not in visited_state:
                # append G to returns (s_t, a_t)
                reward, visits = returns.get((state, action), [0, 0])
                returns[(state, action)] = [reward + G, visits + 1]

                Q[state][action] = (reward + G) / (visits + 1)
                pi[state] = np.argmax(Q[state])  # update our policy greedily

                visited_state.add((state, action))
    v = {state: np.max(values) for state, values in Q.items()}
    return v, pi

def plot_value2(v, eps):
    #v, pi = mc_es(env, generate_eps_from_policy, 0.5, 10000)
    #print(v, '\n')
    #print(pi, '\n')
    # generate_eps(policy, env)
    fig, axes = plt.subplots(nrows=2, subplot_kw={'projection': '3d'})
    fig.suptitle("After %d episodes"%eps, fontsize=16)
    axes[0].set_title('Uable ace')
    axes[1].set_title('No usable ace')
    plot_value(v, axes[0], axes[1])
    plt.show()

def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')

    eps1 = 10000
    eps2 = 500000

    v1 = frist_visit_mc_predict1(policy, env, eps1)
    plot_value2(v1, eps1)

    v2 = frist_visit_mc_predict(policy, env, eps2)
    plot_value2(v2, eps2)

    v3, pi3 = mc_es(env, generate_eps_from_policy, 0.5, eps1)
    plot_value2(v3, eps1)

    #v4, pi4 = mc_es(env, generate_eps_from_policy, 0.5, eps2)
    #plot_value2(v4, eps2)


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