import numpy as np
'''
def recompute_reward_fn_halfcheetahjump(obs, act, next_obs, rew):
    # print('sssssssssssssssssssssssssssss')
    # assert len(obs.shape) == len(next_obs.shape) == len(act.shape)
    # if len(obs.shape) == 1:
    #     new_rew = -(rew + 0.1 * np.sum(np.square(act))) - 0.1 * np.sum(np.square(act))
    #     return new_rew
    # elif len(obs.shape) == 2:
    #     new_rew = -(rew + 0.1 * np.sum(np.square(act), axis=1)) - 0.1 * np.sum(np.square(act), axis=1)
    #     return new_rew
    # else:
    #     raise NotImplementedError
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape)
    if len(obs.shape) == 1:
        new_rew = rew + 5 * obs[0]
        # print(rew,obs[0])
        return new_rew
    elif len(obs.shape) == 2:
        new_rew = rew + 5 * obs[:,0]
        return new_rew

def recompute_reward_fn_antangle(obs, act, next_obs, rew):
    pass

'''