"""
sources:
openai gym frozen lake value iteration: https://learning.oreilly.com/library/view/hands-on-reinforcement-learning/9781788836524/e8ad36d5-21fe-442f-8133-3cee6bf31b2e.xhtml
openai gym frozen lake policy iteration: https://learning.oreilly.com/library/view/hands-on-reinforcement-learning/9781788836524/7aef351a-58cf-4848-9cee-0fd5557e75b1.xhtml
pymdptoolbox hiive fork: https://github.com/hiive/hiivemdptoolbox
mdptoolbox matlab mdp implementation: https://www.mathworks.com/matlabcentral/fileexchange/25786-markov-decision-processes-mdp-toolbox 
"""


import sys
import time

sys.path.append('./hiivemdptoolbox')

from hiive.mdptoolbox import mdp
from hiive.mdptoolbox import example
# from hiive.visualization.mdpviz import 

from pprint import pprint as ppr

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import pandas as pd

import gym


def save_data(stats,name):
    df = pd.DataFrame().from_records(stats)
    df.to_csv('data/{}.csv'.format(name))

def plot_stats(stats,name,y_col='Error'):
    df = pd.DataFrame.from_records(stats)
    # print(df.tail(5))
    plt.clf()
    plt.title('{}, time - {}'.format(name,df['Time'].max()))
    plt.xlabel('Iterations')
    plt.ylabel('Utility Span')
    plt.plot(df.reset_index()['index'],df[y_col])
    plt.tight_layout()
    plt.savefig('plots/{}_{}.png'.format(name,y_col))
    # print(df)

def plot_error(stats,name):
    df = pd.DataFrame.from_records(stats)
    # print(df.tail(5))
    plt.clf()
    plt.title('{} - {}'.format(name,df['Time'].max()))
    plt.xlabel('iterations')
    plt.ylabel('max value')
    plt.plot(df.reset_index()['index'],df['Error'])
    plt.tight_layout()
    plt.savefig('plots/{}.png'.format(name))
    # print(df)

def plot_map(desc,name):
    """
    source: https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values
    """
    # print(desc)

    vals = {
        b'S':0,
        b'F':1,
        b'H':2,
        b'G':3,
    }
    color_vals = {
        0:'springgreen',
        1:'cornflowerblue',
        2:'dimgrey',
        3:'yellow',
    }
    
    fixed = np.array([[vals[col] for col in row] for row in desc])
    
    # print(fixed)
    

    cmap = colors.ListedColormap([color_vals[v] for v in vals.values()])
    bounds = [i - 0.5 for i in range(len(vals)+1)]
    # bounds = [-0.5,0.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # plt.clf()
    plt.figure(figsize=(10,10))
    # plt.title('{}'.format(name))
    plt.imshow(fixed,cmap=cmap,norm=norm)
    plt.grid(which='major',axis='both',linestyle='-',color='k',linewidth=1)
    plt.xticks(np.arange(-.5, fixed.shape[0], 1))
    plt.yticks(np.arange(-.5, fixed.shape[1], 1))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/{}.png'.format(name))

def plot_quiver(policy,prob,name):
    directions = {
        0:np.array([-1,0]),
        1:np.array([0,-1]),
        2:np.array([1,0]),
        3:np.array([0,1]),
    }

    policy = policy[::-1]
    x,y = np.meshgrid(np.arange(0,-int(np.sqrt(policy.shape[0])),-1), np.arange(0,int(np.sqrt(policy.shape[0])),1))
    # x = x - 0.5
    # y = y + 0.5
    z = np.array([directions[action] for i,action in enumerate(policy)])
    # z[0] = np.array([1,1])
    # z[15] = np.array([1,1])
    # print(z)

    # v, u = np.gradient(z[:,0]) , np.gradient(z[:,1])
    # v, u = z[:,0], z[:,1]

    u,v = z[:,0], z[:,1]
    v[v < 0] = -1
    v[v > 0] = 1
    u[u < 0] = -1
    u[u > 0] = 1

    # v,u = np.ones((policy.shape[0],)), np.zeros((policy.shape[0],))
    # print(u)
    # print(v)

    plt.clf()
    plt.quiver(x,y,u,v)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/{}.png'.format(name))

def plot_quiver_map(desc,policy,value,name):
    vals = {
        b'S':0,
        b'F':1,
        b'H':2,
        b'G':3,
    }
    color_vals = {
        0:'springgreen',
        1:'cornflowerblue',
        2:'dimgrey',
        3:'yellow',
    }
    
    fixed = np.array([[vals[col] for col in row] for row in desc])
    policy = np.array(policy)
    # print(fixed)
    # print(policy)
    
    # print(fixed)
    

    cmap = colors.ListedColormap([color_vals[v] for v in vals.values()])
    bounds = [i - 0.5 for i in range(len(vals)+1)]
    # bounds = [-0.5,0.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.clf()
    plt.figure(figsize=(10,10))
    # plt.title('{}'.format(name))
    plt.imshow(fixed,cmap=cmap,norm=norm)
    plt.grid(which='major',axis='both',linestyle='-',color='k',linewidth=1)
    plt.xticks(np.arange(-.5, fixed.shape[0], 1))
    plt.yticks(np.arange(-.5, fixed.shape[1], 1))

    directions = {
        0:np.array([-1,0]),
        1:np.array([0,-1]),
        2:np.array([1,0]),
        3:np.array([0,1]),
    }

    # value = np.array(value)
    # print(value)

    policy = policy[::-1]
    x,y = np.meshgrid(np.arange(0,-int(np.sqrt(policy.shape[0])),-1), np.arange(0,-int(np.sqrt(policy.shape[0])),-1))
    x = x + int(np.sqrt(policy.shape[0]))-1
    y = y + int(np.sqrt(policy.shape[0]))-1

    z = np.array([directions[action] for i,action in enumerate(policy)])


    u,v = z[:,0], z[:,1]
    v[v < 0] = -1
    v[v > 0] = 1
    u[u < 0] = -1
    u[u > 0] = 1

    plt.quiver(x,y,u,v)

    # for i in x:
    #     for j in y:
    #         plt.text(i,j,'hello',horizontalalignment='center',verticalalignment='center')

    # x,y = np.meshgrid(np.arange(0,-int(np.sqrt(value.shape[0])),-1), np.arange(0,-int(np.sqrt(value.shape[0])),-1))


    plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/{}.png'.format(name))

def plot_quiver_map_mdp(policy,name):
    vals = {
        b'S':0,
        b'F':1,
        b'H':2,
        b'G':3,
    }
    color_vals = {
        0:'springgreen',
        1:'cornflowerblue',
        2:'dimgrey',
        3:'yellow',
    }
    
    # fixed = np.array([[vals[col] for col in row] for row in desc])
    fixed = policy
    
    # print(fixed)
    

    cmap = colors.ListedColormap([color_vals[v] for v in vals.values()])
    bounds = [i - 0.5 for i in range(len(vals)+1)]
    # bounds = [-0.5,0.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.clf()
    plt.figure(figsize=(10,10))
    # plt.title('{}'.format(name))
    plt.imshow(fixed,cmap=cmap,norm=norm)
    plt.grid(which='major',axis='both',linestyle='-',color='k',linewidth=1)
    plt.xticks(np.arange(-.5, fixed.shape[0], 1))
    plt.yticks(np.arange(-.5, fixed.shape[1], 1))

    directions = {
        0:np.array([-1,0]),
        1:np.array([0,-1]),
        2:np.array([1,0]),
        3:np.array([0,1]),
    }

    policy = policy[::-1]
    x,y = np.meshgrid(np.arange(0,-int(np.sqrt(policy.shape[1])),-1), np.arange(0,-int(np.sqrt(policy.shape[1])),-1))
    x = x + int(np.sqrt(policy.shape[0]))-1
    y = y + int(np.sqrt(policy.shape[0]))-1
    # x = x - 0.5
    # y = y + 0.5
    z = np.array([directions[action] for i,action in enumerate(policy)])
    # z[0] = np.array([1,1])
    # z[15] = np.array([1,1])
    # print(z)

    # v, u = np.gradient(z[:,0]) , np.gradient(z[:,1])
    # v, u = z[:,0], z[:,1]

    u,v = z[:,0], z[:,1]
    v[v < 0] = -1
    v[v > 0] = 1
    u[u < 0] = -1
    u[u > 0] = 1

    # v,u = np.ones((policy.shape[0],)), np.zeros((policy.shape[0],))
    # print(u)
    # print(v)

    # plt.clf()
    plt.quiver(x,y,u,v)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/{}.png'.format(name))


def run_algo(alg,name,env):
    stats = alg.run()
    # env.render_policy(np.array(alg.policy))
    plot_stats(stats,name)
    plot_quiver_map(env.desc,alg.policy,name)
    ppr(np.array(alg.policy))

def run_forest():
    np.random.seed(0)

    P,R = example.forest(S=4,r1=3,r2=10,p=0.2)
    # print(R)
    gamma = 0.8

    for gamma in [1.0,0.5,0.1]:
        print('VI {}'.format(gamma))
        vi = mdp.ValueIteration(transitions=P,reward=R,gamma=gamma,epsilon=0.000001,max_iter=10000)
        stats = vi.run()
        plot_stats(stats,'vi_forest_{}'.format(gamma))
        print(vi.policy)
        # plot_quiver_map(env.desc,vi.policy,vi.V,'vi_quiver_forest_{}'.format(gamma))
        # print(stats)

    for gamma in [1.0,0.5,0.1]:
        print('PI {}'.format(gamma))
        pi = mdp.PolicyIteration(transitions=P,reward=R,gamma=gamma,max_iter=10000,eval_type=1)
        stats = pi.run()
        plot_stats(stats,'pi_forest_{}'.format(gamma))
        print(pi.policy)
        # plot_quiver_map(env.desc,pi.policy,pi.V,'pi_quiver_forest_{}'.format(gamma))

    # for gamma in [1.0,0.5,0.1]:
    for alpha in [1.0,0.5,0.1]:
        for e in [1.0,0.5,0.1]:
            print('QL: alpha - {} epsilon - {}'.format(alpha,e))
            ql = mdp.QLearning(transitions=P,reward=R,gamma=0.4,alpha=alpha,alpha_decay=1.0,alpha_min=0.001,epsilon=e,epsilon_min=0.1,epsilon_decay=1.0,n_iter=10e5)
            stats = ql.run()
            plot_stats(stats,'ql_forest_{}_{}'.format(alpha,e))
            print(ql.policy)
            # plot_quiver_map(env.desc,ql.policy,ql.V,'ql_quiver_forest_{}_{}'.format(alpha,e))



def run_lake_pr():
    # sys.path.append('./gym_custom')
    sys.path.append('./gym-frozenlake')
    # from gym_custom.gym.envs.toy_text import frozen_lake
    
    import random
    import numpy as np

    import gym
    import gym_frozenlake

    def get_lake_pr(env):
        p = np.zeros((env.nA,env.nS,env.nS,))
        r = np.zeros((env.nA,env.nS,env.nS,))

        for state,actions in env.P.items():
            for action,states_next in actions.items():
                for k,states_next in enumerate(states_next):
                    p[action,state,states_next[1]] = states_next[0]
                    r[action,state,states_next[1]] = states_next[2]

        return p,r


    np.random.seed(0)

    env = gym.make('FrozenLake-v9',random=35,is_slippery=False)
    # env = gym.make('FrozenLake-v9',random=5,is_slippery=True)

    # env.render()
    plot_map(env.desc,'lakemap')

    P,R = get_lake_pr(env)
    # print(P)
    # print(R)

    # gamma = 0.8
    # vi = mdp.ValueIteration(transitions=P,reward=R,gamma=gamma,epsilon=0.1,max_iter=100000)
    # run_algo(vi,'vi_lake_{}'.format(gamma),env)

            



    for gamma in [1.0,0.5,0.1]:
        print('VI {}'.format(gamma))
        vi = mdp.ValueIteration(transitions=P,reward=R,gamma=gamma,epsilon=0.000001,max_iter=10000)
        stats = vi.run()
        plot_stats(stats,'vi_lake_{}'.format(gamma))
        plot_quiver_map(env.desc,vi.policy,vi.V,'vi_quiver_lake_{}'.format(gamma))
        # print(stats)
    
    for gamma in [1.0,0.5,0.1]:
        print('PI {}'.format(gamma))
        pi = mdp.PolicyIteration(transitions=P,reward=R,gamma=gamma,max_iter=10000,eval_type=1)
        stats = pi.run()
        plot_stats(stats,'pi_lake_{}'.format(gamma))
        plot_quiver_map(env.desc,pi.policy,pi.V,'pi_quiver_lake_{}'.format(gamma))

    # for gamma in [1.0,0.5,0.1]:
    for alpha in [1.0,0.5,0.1]:
        for e in [1.0,0.5,0.1]:
            print('QL: alpha - {} epsilon - {}'.format(alpha,e))
            ql = mdp.QLearning(transitions=P,reward=R,gamma=0.4,alpha=alpha,alpha_decay=1.0,alpha_min=0.001,epsilon=e,epsilon_min=0.1,epsilon_decay=1.0,n_iter=10e5)
            stats = ql.run()
            plot_stats(stats,'ql_lake_{}_{}'.format(alpha,e))
            plot_quiver_map(env.desc,ql.policy,ql.V,'ql_quiver_lake_{}_{}'.format(alpha,e))




def run():
    run_forest()
    run_lake_pr()




if __name__ == '__main__':
    run()