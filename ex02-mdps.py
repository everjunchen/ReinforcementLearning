import gym
import numpy as np
import itertools
from gym.envs.toy_text.frozen_lake import generate_random_map

# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
env = gym.make("FrozenLake-v0", desc=custom_map3x3)
''' 
# TODO: Uncomment the following line to try the default map (4x4):
env = gym.make("FrozenLake-v0")
'''

# Uncomment the following lines for even larger maps:
#random_map = generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

r = np.zeros(n_states)  # the r vector is zero everywhere except for the goal state (last state)
r[-1] = 1.

gamma = 0.8

""" This is a helper function that returns the transition probability matrix P for a policy """
def trans_matrix_for_policy(policy):
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions


""" This is a helper function that returns terminal states """
def terminals():
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy):
    P = trans_matrix_for_policy(policy)
    # TODO: calculate and return v
    # (P, r and gamma already given)
    v = np.zeros(n_states)
    idt_matrix = np.identity(P.shape[0])
    inv_matrix = np.linalg.inv(idt_matrix - gamma * P)
    v = inv_matrix.dot(r)
    return v


def bruteforce_policies():
    terms = terminals()
    optimalpolicies = []

    policy = np.zeros(n_states,
                      dtype=np.int)  # in the discrete case a policy is just an array with action = policy[state]
    optimalvalue = np.zeros(n_states)

    # TODO: implement code that tries all possible policies, calculate the values using def value_policy. Find the optimal values and the optimal policies to answer the exercise questions.
    # print('policy shape is ', policy.shape)
    # print('action, states shape is ', n_actions, n_states)

    all_possible_policies = list(map(list, itertools.product(range(n_actions), repeat=n_states)))
    print(len(all_possible_policies))

    counter1 = 0
    counter2 = 0
    counter3 = 0

    for i in all_possible_policies:
        v = value_policy(i)
        #print('policy', i)
        #print('value', v)
        if (np.sum(v) > np.sum(optimalvalue)):
            counter1 += 1
            for j in range(len(optimalpolicies)):
                optimalpolicies.pop()
            optimalpolicies.append(i)
            optimalvalue = v
            print(i)
        elif (np.sum(v) == np.sum(optimalvalue)):
            counter2 += 1
            optimalpolicies.append(i)
            optimalvalue = v
        else:
            counter3 += 1
    '''
    for p in all_possible_policies:
        V = value_policy(p)
        for s in range(n_states):
            a=0
            for a in p:
                for prob, next_state, reward, done in env.P[s][a]:
                    a += prob * (reward + gamma * V[next_state])
             V[s] = a

        if (np.max(V) > optimalvalue):
            counter1 += 1
            for j in range(len(optimalpolicies)):
                optimalpolicies.pop()
            optimalpolicies.append(p)
            optimalvalue = V
            print(p)
        elif (np.max(V) == optimalvalue):
            counter2 += 1
            optimalpolicies.append(p)
        else:
            counter3 += 1

    '''

    print("counter:", counter1,counter2,counter3)

    print("Optimal value function:")
    print(optimalvalue)
    print("number optimal policies:")
    print(len(optimalpolicies))
    print("optimal policies:")
    print(np.array(optimalpolicies))


    return optimalpolicies


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # Here a policy is just an array with the action for a state as element
    policy_left = np.zeros(n_states, dtype=np.int)  # 0 for all states
    policy_right = np.ones(n_states, dtype=np.int) * 2  # 2 for all states

    # Value functions:
    print("Value function for policy_left (always going left):")
    print(value_policy(policy_left))
    print("Value function for policy_right (always going right):")
    print(value_policy(policy_right))

    optimalpolicies = bruteforce_policies()

    # This code can be used to "rollout" a policy in the environment:
    #"""
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(optimalpolicies[0][state])
        env.render()
        state=new_state
        print(reward)
        if done:
            print ("Finished episode")
            break
            #"""


if __name__ == "__main__":
    main()