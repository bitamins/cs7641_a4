from gym.envs.registration import register
 
register(id='FrozenLake-v9', 
    entry_point='gym_frozenlake.envs:FrozenLakeEnv',
    kwargs={'random' : 100},
    # max_episode_steps=10000,
    # reward_threshold=0.99, # optimum = 1
)