import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Deque, Dict, List, Tuple
from replay_buffer import *
from utils import * 
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from preprocess_frame import * 
import copy
from frame_stack import *
import os

class DQNAgent:

    def __init__(
            self,
            memory_size: int = 1024,
            batch_size: int = 32,
            target_update: int = 100,
            gamma: float = 0.99,
            lr: float = 0.001,
            hidden_size = 128,
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            n_step: int = 3,
            plot: bool = False,
            frame_interval: int = 100,
            no_dueling: bool = False,
            no_double: bool = False,
            no_categorical: bool = False,
            no_priority: bool = False,
            no_n_step: bool = False,
            max_epsilon: float = 1.,
            min_epsilon: float = 0.1,
            epsilon_decay: float = 0.0005,
            max_reward: float = None,
            min_reward: float = None,
            frame_preprocess: np.array = None,
            early_stopping: bool = True,
            n_frames_stack: int = 1,
            training_delay: int = 0,
            model_path: str = "models",
            model_name: str = "rainbow"
    ):
        obs_shape = env.observation_space.shape
        if frame_preprocess is not None:
            obs_shape = frame_preprocess(np.zeros(obs_shape)).shape
        if n_frames_stack > 1:
            obs_shape = list(obs_shape)
            obs_shape[0] * = n_frames_stack
        assert len(obs_shape) == 3 or len(obs_shape) == 1
        if len(obs_shape) == 1:
            print("Using DenseNet")
            self.obs_dim = [obs_shape[0]]
            self.mode = "dense"
            self.frame_stack = FrameStack(n_frames_stack, model="array")
        else:
            print("Using ConvNet")
            self.obs_dim = [obs_shape[0], obs_shape[1], obs_shape[2]]
            self.mode = "conv"
            self.frame_stack = FrameStack(n_frames_stack, mode="pixels")
        
        self.action_dim = env.action_space.n
        self.n_frames_stack = n_frames_stack

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.hidden_size = hidden_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Device", self.device)

        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            self.obs_dim, memory_size, batch_size, alpha=alpha, n_step=n_step, gamma=gamma
        )
        self.n_step = n_step

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        if self.mode == "dense":
            self.dqn = DenseNet(
                self.obs_dim[0], self.action_dim, self.atom_size, self.support, self.hidden_size, no_dueling, no_noise
            ).to(self.device)
            self.dqn_target = DenseNet(
                self.obs_dim[0]. self.action_dim, self.atom_size, self.support, self.hidden_size, no_dueling, no_noise
            ).to(self.device)
        else:
            self.dqn = ConvNet(
                self.obs_dim, self.action_dim, self.atom_size, self.support, no_dueling, no_noise
            ).to(self.device)
            self.dqn_target = ConvNet(
                self.obs_dim, self.action_dim, self.atom_size, self.support, no_dueling, no_noise
            ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.early_stopping = early_stopping

        self.lr = lr
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)

        self.transition = list()

        self.training_delay = training_delay
        
        self.is_test = False

        self.max_reward = max_reward
        self.min_reward = min_reward

        self.model_dir = model_path
        self.model_path = os.path.join(model_path, model_name, + ".tar")

        self.frame_preprocess = frame_preprocess

        self.plot = plot
        self.frame_interval = frame_interval

        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = self.max_epsilon
        self.epsilon_decay = epsilon_decay

        self.no_dueling = no_dueling
        self.no_double = no_double
        if no_double:
            self.dqn_target = self.dqn

        self.no_noise = no_noise
        if no_noise:
            self.epsilon, self.max_epsilon, self.min_epsilon = 0, 0, 0
        self.no_categorical = no_categorical
        self.no_n_step = no_n_step
        if no_n_step:
            self.n_step = 1
            self.memory = PrioritizedReplayBuffer(
                self.obs_dim, memory_size, batch_size, alpha=alpha, n_step=self.n_step, gamma = gamma
            )
        self.no_priority = no_priority
        if no_priority:
            self.alpha = 0
            self.memory = PrioritizedReplayBuffer(
                self.obs_dim, memory_size, batch_size, alpha=alpha, n_step=self.n_step, gamma=gamma
            )
    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.no_noise and self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()

        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)
        if self.frame_preprocess is not None:
            next_state = self.frame_preprocess(next_state)
        if self.n_frames_stack > 1:
            next_state = self.get_n_frames(next_state)

        if self.max_reward is not None:
            if reward > self.max_reward:
                reward = self.max_reward
        
        if self.min_reward is not None:
            if reward < self.min_reward:
                reward = self.min_reward

        if no self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done
    
    def update_model(self) -> torch.Tensor:
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(sample["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        gamma = self.gamma ** self.n_step
        elemenwise_loss = self._compute_dqn_loss(samples, gamma)

        loss = torch.mean(elemenwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        loss_for_prior = elemenwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)


        if no self.no_noise:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()
        
        return loss.item()
    
    def train(self, num_frames: int) -> (List[int], List[int]):
        self.is_test = False

        state = self.env.reset()

        state = self.init_first_frame(state)

        update_cnt = 0
        losses = []
        scores = []
        frame_scores = []
        score = 0
        if self.early_stopping: 
            best_model = copy.deepcopy(self.dqn.state_dict())
            best_average_score = -np.inf

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            self.update_beta()

            if done:
                scores.append(score)
                state = self.env.reset()
                state = self.init_first_frame(state)
                score = 0

            if self.no_noise:
                self.set_epsilon()

            if len(self.memory) >= self.batch_size and self.training_delay <= 0:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                if update_cnt % self.target_update == 0 and not self.no_doulbe:
                    self._target_hard_update()

                if frame_idx % self.frame_interval == 0:
                    if len(scores) == 0:
                        if len(frame_scores) > 0:
                            frame_scores.append(float(frame_score[-1]))
                        else:
                            frame_scores.append(0.)
                    
                    else:
                        frame_scores.append(float(np.mean(scores)))

                    if self.plot:
                        self._plot(frame_idx, frame_scores, losses)
                    
                    scores = []

                    if self.early_stopping and frame_scores[-1] > best_average_score:
                        best_average_score = frame_scores[-1]
                        best_model = copy.deepcopy(self.dqn.state_dict())
                    
                    self.save()

                if self.training_delay > 0:
                    self.training_delay -= 1
                
            if self.early_stopping:
                self.dqn_load_state_dict(best_model)
            self.env.close()

            return frame_scores, losses
        
        def test(self, get_frames=False, get_actions=False) -> (int, List[int]) or (int, List[np.ndaaray]):
            self.is_test = True

            state = self.env.reset()
            state = self.init_first_frame(state)

            done = False
            score = 0

            actions = []
            frames = []

            while not done: 
                self.env.render()
                action = self.select_action(state)
                if get_actions:
                    actions.append(actions)
                next_state, reward, done = self.step(actions)

                state = next_state
                score += reward

                if get_frames: 
                    frames.append(self.env.render(mode='rgb_array'))

            self.env.close()

            if get_frames and not get_actions:
                return score, frames
            if not get_frames and get_actions:
                return score, actions
            if get_frames and get_actions:
                return score, frames, actions
            
            return score
        
        def _compute_dqn_loss(self, samples: Dict[str, np.ndaaray], gamma: float) -> torch.Tensor:
            device = self.device
            state = torch.FloatTensor(samples["obs"]).to(device)
            next_state = torch.FloatTensor(samples["next_obs"]).to(device)
            reward = torch.FloaTensor(samples["rews"].reshape(-1, 1)).to(device)
            done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

            if not self.no_categorical:
                delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

                with torch.no_grad():
                    next_action = self.dqn(next_state).argmax(1)
                    next_dist = self.dqn_target.dist(next_state)
                    next_dist = next_dist[range(self.batch_size), next_action]

                    t_z = reward + (1 - done) * gamma * self.support
                    t_z = t_z.clamp(min=self.v_min, max=self.v_max)
                    b = (t_z - self.v_min) / delta_z
                    l = b.floor().long()
                    u = b.ceil().long()

                    offset = (
                        torch.linspace(
                            0, (self.batch_size - 1) * self.atom_size, self.batch_size
                        ).long().unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device)
                    )

                    proj_dist = torch.zeros(next_dist.size(), device=self.device)
                    proj_dist.view(-1).index_add_(
                        0, (1+ offset).view(-1), (next_dist * (u.float() -b)).view(-1)
                    )
                    proj_dist.view(-1).index_add_(
                        0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                    )

                    dist = self.dqn.(state)
                    log_p = torch.log(dist[range(self.batch_size), action])
                    loss = -(proj_dist * log_p).sum(1)

            else:
                #456