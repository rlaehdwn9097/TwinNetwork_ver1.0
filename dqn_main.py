# DQN main
# coded by St.Watermelon

from dqn_learn import DQNagent
import config as cf

def main():
    max_episode_num = cf.MAX_EPISODE_NUM
    agent = DQNagent()
    agent.train(max_episode_num)
    agent.write_meta_file()
    agent.plot_result()
    

if __name__=="__main__": 
    main()
    