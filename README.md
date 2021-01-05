# Reinforcement Learning - Lunar Lander practical samples

In this repository, we explore a couple of practical approaches for solving OpenAI's LunarLander-v2 gym [https://gym.openai.com/envs/LunarLander-v2/](https://gym.openai.com/envs/LunarLander-v2/).

<!-- blank line -->
<figure class="video_container">
  <video controls="true" allowfullscreen="false">
    <source src="media/original.mp4" type="video/mp4">
  </video>
</figure>
<!-- blank line -->

**Content:**

**[1. Lunar Lander workshop for Amazon SageMaker (AWS)](./AWS-LunarLander-workshop)**

This content is inspired on the [AWS Lunar Lander workshop](https://lunarlander.ninja/), where you can gamify the experience of using Reinforcement Learning and the PPO algorithm.

For running this excercise you must create an Amazon SageMaker notebook, clone this repository on it, and execute the two main notebooks included in this order:
- Intro to Gym and Lunar Lander
- lunarlander

*Note: There is an issue with the rendering of the simulations in the videos. Troubleshooting ongoing*.

**[2. Lunar Lander excercise for Colab (Google Colab)](./ngoodger-LunarLander-PPO)**

This content includes two notebooks for covering the recurrent training with PPO on the popular OpenAI gyms.
For running this excercise you must upload the notebooks to a Google Colab environment, and execute these in this order:

- Recurrent PPO: For setting up the gym and training a RL model based on PPO
- Test recurrent PPO: For evaluating the checkpoints trained and visualizing the results of the simulations

---

