import numpy as np
import random
import pickle

width = 0.25

def generate_toy_state():
    while True:
        x = random.uniform(-3, 3)  
        y = random.uniform(-3, 3) 
        if -x - width < y < -x + width: 
            return (x, y)  
       
def generate_toy_action():
     x = random.uniform(-1, 1)
     y = random.uniform(-1, 1)
     return (x,y)

def calculate_reward(pos):
     line_distance = abs(pos[1] + pos[0]) / np.sqrt(2)

     return line_distance if pos[1] > -pos[0] else -line_distance
       

def generate_dataset(num_samples):
     obs_ = []
     next_obs_ = []
     action_ = []
     reward_ = []
     done_ = []
     episode_step = 0
     
     for _ in range(num_samples):
          obs = generate_toy_state()
          action = generate_toy_action()
          new_obs = (obs[0]+action[0],obs[1]+action[1])
          reward = calculate_reward(new_obs)
          done_bool = False
          
          
          obs_.append(obs)
          next_obs_.append(new_obs)
          action_.append(action)
          reward_.append(reward)
          done_.append(done_bool)
          episode_step += 1
          
     dataset =  {
          'observations': np.array(obs_),
          'actions': np.array(action_),
          'next_observations': np.array(next_obs_),
          'rewards': np.array(reward_),
          'terminals': np.array(done_),
     }

     return dataset

if __name__=="__main__":
     
     dataset = generate_dataset(10000)
     with open('./random_dataset_0.25width_10000.pkl', 'wb') as file:
          pickle.dump(dataset, file)
     file.close()