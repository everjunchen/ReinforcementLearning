import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def print_policy(Q, env):
    """ This is a helper function to print a nice policy from the Q function"""
    moves = [u'←', u'↓', u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in ['H', 'G']:
            policy[idx] = u'·'
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row])
                     for row in policy]))


def plot_V(Q, env):
    """ This is a helper function to plot the state values from the Q function"""
    fig = plt.figure()
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
        if env.desc[idx] in ['H', 'G']:
            V[idx] = 0.
    plt.imshow(V, origin='upper',
               extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6,
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y + 0.5, dims[0] - x - 0.5, '{:.3f}'.format(V[x, y]),
                 horizontalalignment='center',
                 verticalalignment='center')
    plt.xticks([])
    plt.yticks([])


def plot_Q(Q, env):
    """ This is a helper function to plot the Q function """
    from matplotlib import colors, patches
    fig = plt.figure()
    ax = fig.gca()

    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape

    up = np.array([[0, 1], [0.5, 0.5], [1, 1]])
    down = np.array([[0, 0], [0.5, 0.5], [1, 0]])
    left = np.array([[0, 0], [0.5, 0.5], [0, 1]])
    right = np.array([[1, 0], [0.5, 0.5], [1, 1]])
    tri = [left, down, right, up]
    pos = [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5], [0.5, 0.8]]

    cmap = plt.cm.RdYlGn
    norm = colors.Normalize(vmin=.0, vmax=.6)

    ax.imshow(np.zeros(dims), origin='upper', extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6, cmap=cmap)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        x, y = idx
        if env.desc[idx] in ['H', 'G']:
            ax.add_patch(patches.Rectangle((y, 3 - x), 1, 1, color=cmap(.0)))
            plt.text(y + 0.5, dims[0] - x - 0.5, '{:.2f}'.format(.0),
                     horizontalalignment='center',
                     verticalalignment='center')
            continue
        for a in range(len(tri)):
            ax.add_patch(patches.Polygon(tri[a] + np.array([y, 3 - x]), color=cmap(Q[s][a])))
            plt.text(y + pos[a][0], dims[0] - 1 - x + pos[a][1], '{:.2f}'.format(Q[s][a]),
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=9, fontweight=('bold' if Q[s][a] == np.max(Q[s]) else 'normal'))

    plt.xticks([])
    plt.yticks([])

def plot_eps_length(average_eps_length):
    fig = plt.figure()
    ax = fig.add_subplot()
    a = np.arange(0, len(average_eps_length))
    ax.plot(a,average_eps_length)
    ax.set_xlabel('episode')
    ax.set_ylabel('average episode length')


def epsilon_greedy_action(env, epsilon, Q, state):
    if np.random.uniform(0, 1) < epsilon:
        a = np.random.randint(env.action_space.n)  #
    else:
        a = np.argmax(Q[state, :])
    return a

def sarsa(env, alpha=0.35, gamma=0.9, epsilon=0.8, num_ep=int(1e4)):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # TODO: implement the sarsa algorithm
    eps_length = []
    average_eps_length = []
    # This is some starting point performing random walks in the environment:
    for i in range(num_ep):
        s = env.reset()
        a = epsilon_greedy_action(env, epsilon, Q, s)
        done = False
        count_steps = 0
        while not done:
            #a = np.random.randint(env.action_space.n)
            s_, r, done, _ = env.step(a)
            a_ = epsilon_greedy_action(env, epsilon, Q, s_)
            Q[s][a] += alpha*(r + gamma*Q[s_][a_]-Q[s][a])
            s = s_
            a = a_
            count_steps += 1
        eps_length.append(count_steps)
        average_eps_length.append(np.sum(eps_length)/(i+1))
        #print('episode:',i, eps_length[i], average_eps_length[i])

    return Q, average_eps_length


def qlearning(env, alpha=0.35, gamma=0.9, epsilon=0.8, num_ep=int(1e4)):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # TODO: implement the qlearning algorithm
    eps_length = []
    for i in range(num_ep):
        s = env.reset()
        done = False
        count_steps = 0
        while not done:
            #a = np.random.randint
            a = epsilon_greedy_action(env, epsilon, Q, s)
            s_, r, done, _ = env.step(a)
            #a_ = np.argmax(Q[s_, :])
            #Q[s][a] += alpha * (r + gamma * Q[s_][a_] - Q[s][a])
            Q[s][a] += alpha*(r + gamma*np.max(Q[s_,:])-Q[s][a])
            s = s_
            count_steps += 1
        eps_length.append(count_steps)
        #print('episode:',i, np.sum(eps_length)/(i+1))
    return Q


#env = gym.make('FrozenLake-v0')
#env =gym.make('FrozenLake-v0', is_slippery=False)
env=gym.make('FrozenLake-v0', map_name="8x8")
#check states
print("current environment: ")
env.render()
print("")

#'''
print("Running sarsa...")
Q, eps_length = sarsa(env)
plot_eps_length(eps_length)
plot_V(Q, env)
plot_Q(Q, env)
print_policy(Q, env)
plt.show()

print("Running qlearning")
Q = qlearning(env)
plot_V(Q, env)
plot_Q(Q, env)
print_policy(Q, env)
plt.show()#'''