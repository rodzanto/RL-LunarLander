import json
import os

import gym
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from sagemaker_rl.ray_launcher import SageMakerRayLauncher


def create_environment(env_config):
    # This import must happen inside the method so that worker processes import this code
    return gym.make('LunarLander-v2')


class MyLauncher(SageMakerRayLauncher):

    def register_env_creator(self):
        register_env("LunarLander-v2", create_environment)
        
    #This is where you edit the hyperparameters. Change the values in the config section listed below.
    #The hyperparameters we recommend on changing are: lambda, kl_coeff, vf_loss__coeff, num_sgd_iter, clip_param
    #Check the notebook Step 6 for tips on tuning hyperparameters. 
    def get_experiment_config(self):
        return {
          "training": { 
            "env": "LunarLander-v2",
            "run": "PPO",
            "stop": {
                "episode_reward_mean": 0,
                "training_iteration": 50, #make training iteration smaller to lower training times
            },
            "config": {
              "num_sgd_iter": 330,
              "lr": 5e-5,
              "lambda":0.9998201362028988,
              "vf_loss_coeff":0.9624532314793395,
              "kl_target":0.05500572866402965,
              "kl_coeff":0.22702622027852548,
              "entropy_coeff":0.4906464378184655,
              "clip_param":0.745631860794536,
              "sgd_minibatch_size": 5000,
              "train_batch_size": 25000,
              "monitor": True,  # Record videos.
              "model": {
                "free_log_std": True
              },
              "num_workers": (self.num_cpus-1),
              "num_gpus": self.num_gpus,
              "batch_mode": "complete_episodes",
            }
          }
        }

if __name__ == "__main__":
    MyLauncher().train_main()
    
