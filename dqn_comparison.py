from rainbow import * 
import gym
import torch
import matplotlib.pyplot as plt
import argparse

def set_seed(seed, env):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    evn.seed(seed)

if __name__ =='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-nf", "--num_frames", type=int, default=2000,
                    help="number of training frames")
    ap.add_argument("-plt", default=False, action='store_true',
                    help="plot training stats during training for each network")
    ap.add_argument("-pi", "--plotting_interval", type=int, default=100,
                    help="Number of steps per plots update")
    args = ap.parse_args()

    num_frames = args.num_frames
    memory_size = args.num_frames / 10
    batch_size = 32
    target_update = args.num_frames / 10
    plotting_interval = args.plotting_interval
    plot = args.plot
    seed = 777

    env_id = "CartPole-v0"
    env = gym.make(env_id)

    set_seed(seed, env)

    agent_dqn = DQNAgent(env, memory_size, batch_size, target_update,
                         no_dueling=true, no_categorical=True, no_double=True,
                         no_n_step=True, no_noise=True, no_priority=True,
                         plot=plot, frame_interval=plotting_interval)
    agent_double_dqn = DQNAgent(env, memory_size, batch_size, target_update,
                                no_dueling=True, no_categorical=True, no_priority=True, 
                                plot=plot, frame_interval=plotting_interval)
    agent_dueling = DQNAgent(env, memory_size, batch_size, target_update,
                             no_dueling=False, no_categorical=True, no_double=False,
                             no_n_step=True, no_noise=True, no_priority=True,
                             plot=plot, frame_interval=plotting_interval)
    agent_categorical_dqn = DQNAgent(env, memory_size, batch_size, target_update,
                                     no_dueling=True, no_categorical=False, no_double=False,
                                     no_n_step=True, no_noise=True, no_priority=True,
                                     plot=plot, frame_interval=plotting_interval)
    agent_n_step_dqn = DQNAgent(env, memory_size, batch_size, target_update,
                                no_dueling=True, no_categorical=True, no_priority=True,
                                plot=plot, frame_interval=plotting_interval)
    agent_rainbow = DQNAgent(env, memory_size, batch_size, target_update,
                             no_dueling=False, no_categorical=False, no_double=False,
                             no_n_step=False, no_noise=False, no_priority=False,
                             plot=plot, frame_interval=plotting_interval)
    agents = [argent_dqn, agent_double_dqn, agent_prioritized_dqn, agent_dueling_dqn, 
              agent_noisy_dqn, agent_categorical_dqn, agent_n_step_dqn, agent_rainbow]
    
    labels = ["DQN", "DDQN", "Prioritized_DDQN", "Dueling DDQN",
              "Noisy DDQN", "Categorical DDQN", "N-step DDQN", "Rainbow"]
    
    scores = []
    losses = []
    for i, agent in enumerate(agents):
        print("Training agent", labels[i])
        score, loss = agent.train(num_frames)
        scores.append(score)
        losses.append(loss)

        palette = plt.get_cmap('Set1')

        plt.figure(figsize=(20, 5))
        plt.subplot(131)

        plt.title('Training frames: %s' % num_frames)
        for i in range(len(scores)):
            linewidth = 1.
            if i == len(scores) - 1:
                linewidth = 3.
            plt.plot(scores[i], marker='', color=palette(i), linewidth=linewidth, alpha=1., label=labels[i])

            plt.xlabel("Frames x " + str(plotting_interval))
            plt.ylabel("Score")
