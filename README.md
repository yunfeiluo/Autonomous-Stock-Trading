# Autonomous-Stock-Trading
This is the official repository for the workshop paper "Agent Performing Autonomous Stock Trading under Good and Bad Situations" by authors [Yunfei Luo](https://yunfeiluo.github.io/) and [Zhangqi Duan](https://www.linkedin.com/in/zhangqi-duan-4140311b6/) in [AI4ABM](https://ai4abm.org/) at [ICLR 2023](https://iclr.cc/Conferences/2023). 

### Abstract
Stock Trading is one of the popular ways for financial management. However, the market and the environment of economy is unstable and usually not predictable. Furthermore, engaging in stock trading requires time and effort to analyze, create strategies, and make decisions. It would be convenient and effective if an agent could assist or even do the task of analyzing and modeling the past data and then generate a strategy for autonomous trading. Recently, reinforcement learning has been shown to be robust in various tasks that involve achieving a goal with a decision making strategy based on time-series data. In this project, we have developed a pipeline that simulates the stock trading environment and have trained an agent to automate the stock trading process with deep reinforcement learning methods, including deep Q-learning, deep SARSA, and policy gradient method. We evaluate our platform during relatively good (before 2021) and bad (2021 - 2022) situations. The stocks we've evaluated on including Google, Apple, Tesla, Meta, Microsoft, and IBM. These stocks are among the popular ones, and the changes in trends are representative in terms of having good and bad situations. 
We showed that before 2021, the three reinforcement methods we have tried always provide promising profit returns with total annual rates around 70% to 90%, while maintain a positive profit return after 2021 with total annual rates around 2% to 7%. 

---

### Main file
- run.py is the main file for running the training and testing pipeline with methods of deep q-learning, deep SARSA, and policy gradient. 
- The stock that is used for running the pipeline can be set in data_preprocessing.py
