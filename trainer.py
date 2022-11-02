
import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DRL4TSP, Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


class StateCritic(nn.Module):


    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)


        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output




def validate(data_loader, actor, reward_fn, render_fn=None,render_static_fn=None, save_dir='.',
             num_plot=5):

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    exec_time = 0
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            before = time.time()
            tour_indices, logp= actor.forward(static, dynamic, x0)
            after = time.time()
            exec_time += after - before
            dynamic = actor.finaldynamic


        dynamic = dynamic.to(device)
        # print('tour_indices is:', tour_indices)
        reward = reward_fn(dynamic, tour_indices).mean().item()
        # print('The reward is:', reward)
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'dynamic_batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(dynamic, tour_indices, path)

        if render_static_fn is not None and batch_idx < num_plot:
            name2 = 'static_batch%d_%2.4f.png' % (batch_idx, reward)
            path2 = os.path.join(save_dir, name2)
            render_static_fn(static, tour_indices, path2)

    print("execution time for 100 iterations:", exec_time)

    actor.train()
    return np.mean(rewards)


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn,render_static_fn,batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_reward = np.inf

    for epoch in range(80):
        print('This is epoch:', epoch)
        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):
            # print('Now the current batch_idx is:', batch_idx)
            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None


            tour_indices, tour_logp= actor(static, dynamic, x0)
            # print('Now the tour_indices is:,', tour_indices)

            reward = reward_fn(dynamic, tour_indices)
            # print('The reward is:', reward)
            critic_est = critic(static, dynamic).view(-1)
            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end
                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)


        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)


        valid_dir = os.path.join(save_dir, '%s' % epoch)
        mean_valid = validate(valid_loader, actor, reward_fn, render_fn, render_static_fn, valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
              np.mean(times)))





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='tsp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', default=5e-5, type=float)
    parser.add_argument('--critic_lr', default=5e-5, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=4000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)

    args = parser.parse_args()

    from tasks import tsp
    from tasks.tsp import TSPDataset

    STATIC_SIZE = 2
    DYNAMIC_SIZE = 3
    train_data = TSPDataset(args.num_nodes, args.train_size, args.seed)
    valid_data = TSPDataset(args.num_nodes, args.valid_size, args.seed + 1)
    update_fn = train_data.update_dynamic

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    update_fn,
                    tsp.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)
    print('Create actor and critic successfully')

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tsp.reward
    kwargs['render_fn'] = tsp.render



    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        print('Now we begin to train the actor')
        train(actor, critic, render_static_fn=tsp.render_static, **kwargs)

    test_data_size = 100
    test_data = TSPDataset(args.num_nodes, test_data_size, args.seed +10)
    original_location = test_data.static
    final_location = test_data.dynamic

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, tsp.reward, tsp.render, tsp.render_static, test_dir, num_plot=5)

    print('Average tour length: ', out)
