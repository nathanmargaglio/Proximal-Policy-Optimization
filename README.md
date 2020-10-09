# Proximal-Policy-Optimization

An implementation of [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) using TensorFlow.

This repository is quite messy, but it contains code which successfully trains an [Ant agent](https://gym.openai.com/envs/Ant-v2/) to walk in a direction as fast as possible in an environment simulated using [MuJoCo](http://www.mujoco.org/).  The code is setup to run in a headless environment (the agent was trained in an AWS EC2 instance), which required quite a bit of tweaking to make work.

![https://github.com/nathanmargaglio/Proximal-Policy-Optimization/blob/master/ppo_ant_long.gif](ppo_ant_short.gif)

It's not pretty, but it works!
