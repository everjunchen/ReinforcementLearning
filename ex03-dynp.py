import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
#env=gym.make('FrozenLake-v0', map_name="8x8")
# you can set it to deterministic with:
#env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
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
    return policy


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break
    """

if __name__ == "__main__":
    main()