import torch
import gym
import time

import numpy as np
import matplotlib.pyplot as plt

from config import agents_dict
from Components.utils import argsparser
from Components.running_stat import ZFilter
from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.utils import NormalizedEnv

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    seed = args.seed
    args.env = "UR5-Reacher"
    print(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create UR5 Reacher2D environment
    env = ReacherEnv(
            setup="UR5_default",
            host='129.128.159.210',
            dof=2,
            control_type="velocity",
            target_type="position",
            reset_type="zero",
            reward_type="precision",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=0.3,
            speedj_a=1.4,
            episode_length_time=4.0,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=0.04,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=np.random.get_state(),
        )
    env = NormalizedEnv(env)
    # Start environment processes
    env.start()

    o_dim = env.observation_space.shape[0]
    a_dim= env.action_space.shape[0]


    # Set the agent
    network = agents_dict[args.agent]
    agent = network(lr=args.lr, gamma=args.gamma, BS=args.batch_size, 
                    o_dim=o_dim, n_actions=a_dim, hidden=args.hidden, args=args,
                    device=device, continuous=True)


    # Experiment block starts
    # Create the buffer
    agent.create_buffer(env)

    ret = step = 0
    rets = []
    all_rets = []
    ep_lens = []
    avgrets = []
    losses = []
    avglos = []
    op = env.reset()


    num_steps = 150000
    checkpoint = 2000
    num_episode = 0
    count = 0
    time = 0
    for steps in range(num_steps):
        # does torch need expand_dims
        a, lprob = agent.act(op)
        if args.continuous:    
            scaled_ac = env.action_space.low + (a + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
            scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
            obs, r, done, infos = env.step(scaled_ac)
        else:
            obs, r, done, infos = env.step(a)
        agent.store(op,r,done,a,lprob,time)

        # Observe        
        op = obs
        time += 1
        count += 1
        
        loss, count = agent.learn(count, obs)
        losses.append(loss)

        # End of Episode
        # ret += r * args.gamma**time
        # For convience
        ret += r
        step += 1
        if done:
            num_episode += 1
            rets.append(ret)
            all_rets.append(ret)
            ep_lens.append(step)
            print("Episode {} ended with return {:.2f} in {} steps. Total steps: {}".format(num_episode, ret, step, steps))

            ret = 0
            step = 0
            time = 0            
            op = env.reset()

        if (steps + 1) % checkpoint == 0:
            # plt.rcParams["figure.figsize"]
            # gymdisplay(env,MAIN)
            avgrets.append(np.mean(rets))
            avglos.append(np.mean(losses))
            rets = []
            losses = []
            plt.clf()
            plt.subplot(211)
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            plt.subplot(212)
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avglos)
            # plt.savefig('Hopper_hyper_graph/hopper_ppo_lr_' + floatToString(args.lr) + "_seed_" + str(
            #     args.seed) + "_agent_" + str(args.agent)  + "_var_" + floatToString(args.var))
            plt.pause(0.001)
            data = np.zeros((2, len(all_rets)))
            data[0, :] = np.array(ep_lens)
            data[1, :] = np.array(all_rets)
            np.savetxt('UR5Reacher_{}_{}_{}.txt'.format(args.agent, args.naive, seed), data)
    env.close()
    env.terminate()
    return avgrets
    
if __name__ == "__main__":
    args = argsparser()
    tic = time.time()
    avgrets = train(args)
    print("Run took:", time.time() - tic)
    np.savetxt("./results/{}_{}_{}-{}.txt".format(args.env, args.agent, args.naive, args.seed), avgrets)
