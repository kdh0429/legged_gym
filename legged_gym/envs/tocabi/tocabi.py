# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .tocabi_config import TOCABIRoughCfg
from legged_gym.utils.math import cubic

class Tocabi(LeggedRobot):
    cfg : TOCABIRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self._clip_actions(actions)
        self._generate_reference_motion()
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self._process_dof_noise()
        #time update
        self.time += self.dt_policy
        self.time += 5*self.dt_policy*self.actions[:,-1].unsqueeze(-1)
        self.post_physics_step()
        self.epi_len[:] += 1

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.obs_history[env_ids,:] = 0
        self.action_history[env_ids,:] = 0


    def compute_observations(self):
        pi =  3.14159265358979 
        q = self.root_states[:,3:7].clone()
        fixed_angle_x, fixed_angle_y, fixed_angle_z = quat2euler(q)

        fixed_angle_x += self.quat_bias[:,0]
        fixed_angle_y += self.quat_bias[:,1]
        fixed_angle_z += self.quat_bias[:,2]
        
        time2idx = (self.time % self.mocap_cycle_period) / self.mocap_cycle_dt
        phase = (self.init_mocap_data_idx + time2idx) % self.mocap_data_num / self.mocap_data_num
        
        vel_noise = torch.rand(self.num_envs, 6, device=self.device, dtype=torch.float)*0.05-0.025
        obs = torch.cat((fixed_angle_x.unsqueeze(-1), fixed_angle_y.unsqueeze(-1), fixed_angle_z.unsqueeze(-1), 
                self.qpos_noise[:,0:12]+self.qpos_bias, 
                self.qvel_noise[:,0:12],
                torch.sin(2*pi*phase) .view(-1,1),
                torch.cos(2*pi*phase).view(-1,1),
                self.commands[:,0].unsqueeze(-1),
                self.commands[:,1].unsqueeze(-1),
                self.root_states[:,7:]+vel_noise),dim=-1)
        
        diff = obs-self.obs_mean
        normed_obs =  diff/torch.sqrt(self.obs_var + 1e-8*torch.ones_like(self.obs_var))

        self.obs_history = torch.cat((self.obs_history[:,self.num_single_step_obs:], normed_obs), dim=-1)
        self.action_history = torch.cat((self.action_history[:,self.num_actions:], self.actions),dim=-1)

        epi_start_idx = (self.epi_len == 0)
        for i in range(self.num_obs_hist*self.num_obs_skip):
            self.obs_history[epi_start_idx,self.num_single_step_obs*i:self.num_single_step_obs*(i+1)] = normed_obs[epi_start_idx,:]

        for i in range(0, self.num_obs_hist):
            self.obs_buf[:,self.num_single_step_obs*i:self.num_single_step_obs*(i+1)] = \
                self.obs_history[:,self.num_single_step_obs*(self.num_obs_skip*(i+1)-1):self.num_single_step_obs*(self.num_obs_skip*(i+1))]

        action_start_idx = self.num_single_step_obs*self.num_obs_hist
        for i in range(self.num_obs_hist-1):
            self.obs_buf[:,action_start_idx+self.num_actions*i:action_start_idx+self.num_actions*(i+1)] = \
                self.action_history[:,self.num_actions*(self.num_obs_skip*(i+1)):self.num_actions*(self.num_obs_skip*(i+1)+1)]

    def _init_buffers(self):
        # Gimmick: Temporally set num_actions to num_dofs for self.p_gains and self.d_gains dimmension matching in LeggedRobot class and set manually again
        self.num_actions = self.num_dofs 
        super()._init_buffers()

        self.num_actuator_action = self.cfg.env.num_actuator_action
        self.num_obs_hist = self.cfg.env.num_obs_hist
        self.num_obs_skip = self.cfg.env.num_obs_skip
        self.num_single_step_obs = self.cfg.env.num_single_step_obs
        self.num_actions = self.cfg.env.num_actions

        self.actuator_high = to_torch(self.cfg.control.actuator_scale, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            for dof_name in self.cfg.control.p_gains.keys():
                if dof_name in self.dof_names[i]:
                    self.p_gains[i] = self.cfg.control.p_gains[dof_name] / 9.0
                    self.d_gains[i] = self.cfg.control.d_gains[dof_name] / 3.0

        self.epi_len = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.epi_len_log = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.dt_sim = self.cfg.sim.dt
        self.dt_policy = self.dt_sim * self.cfg.control.decimation

        self.obs_history = torch.zeros(self.num_envs, self.num_obs_hist*self.num_obs_skip*self.num_single_step_obs, dtype=torch.float, requires_grad=False, device=self.device)
        self.action_history = torch.zeros(self.num_envs, self.num_obs_hist*self.num_obs_skip*self.num_actions, dtype=torch.float, requires_grad=False, device=self.device)

        #for Deep Mimic
        self.init_mocap_data_idx = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.long)
        self.mocap_data_idx = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.long)
        mocap_data_non_torch = np.genfromtxt(self.cfg.motion.file, encoding='ascii') 
        self.mocap_data = torch.tensor(mocap_data_non_torch, device=self.device, dtype=torch.float)
        self.mocap_data_num = int(self.mocap_data.shape[0] - 1)
        self.mocap_cycle_dt = 0.0005
        self.mocap_cycle_period = self.mocap_data_num * self.mocap_cycle_dt
        self.time = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)

        #for observation 
        self.qpos_noise = torch.zeros_like(self.dof_pos)
        self.qvel_noise = torch.zeros_like(self.dof_vel)
        self.qpos_pre = torch.zeros_like(self.dof_pos)   

        self.qpos_bias = torch.rand(self.num_envs, 12, device=self.device, dtype=torch.float)*6.28/100-3.14/100
        self.quat_bias = torch.rand(self.num_envs, 3, device=self.device, dtype=torch.float)*6.28/150-3.14/150

        obs_mean_non_torch = np.genfromtxt(self.cfg.normalization.obs_mean_dir,encoding='ascii')
        obs_var_non_torch = np.genfromtxt(self.cfg.normalization.obs_var_dir,encoding='ascii')
        self.obs_mean = torch.tensor(obs_mean_non_torch, device=self.device, dtype=torch.float)
        self.obs_var = torch.tensor(obs_var_non_torch, device=self.device, dtype=torch.float)

    def _create_envs(self):
        super()._create_envs()

        start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)

        for i in range(self.num_envs):
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    self.envs[i], self.actor_handles[i], j, gymapi.MESH_VISUAL, gymapi.Vec3(0.85938, 0.07813, 0.23438))

            dof_prop = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
            dof_prop['damping'] = len(dof_prop['damping']) * [0.1]
            dof_prop['armature'] = [0.614, 0.862, 1.09, 1.09, 1.09, 0.360,\
                                    0.614, 0.862, 1.09, 1.09, 1.09, 0.360,\
                                    0.078, 0.078, 0.078, \
                                    0.18, 0.18, 0.18, 0.18, 0.0032, 0.0032, 0.0032, 0.0032, \
                                    0.0032, 0.0032, \
                                    0.18, 0.18, 0.18, 0.18, 0.0032, 0.0032, 0.0032, 0.0032]
            dof_prop['velocity'] = len(dof_prop['velocity']) * [4.03]
            self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], dof_prop)


    def _compute_torques(self):
        torques_lower = self.actions[:,0:self.num_actuator_action] * self.actuator_high[:self.num_actuator_action]
        torques_upper = self.p_gains[self.num_actuator_action:]*(self.target_data_qpos[:,self.num_actuator_action:] - self.dof_pos[:,self.num_actuator_action:]) + \
                        self.d_gains[self.num_actuator_action:]*(-self.dof_vel[:,self.num_actuator_action:])
        whole_torque = self.p_gains*(self.default_dof_pos - self.dof_pos) + self.d_gains*(-self.dof_vel)
        
        # return torch.cat([torques_lower, torques_upper], dim=1)
        return whole_torque

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _clip_actions(self, actions):
        self.actions[:,0:self.num_actuator_action] = torch.clip(actions[:,0:self.num_actuator_action], 
                                                                self.cfg.normalization.clip_actuator_actions_low, self.cfg.normalization.clip_actuator_actions_high).to(self.device)
        self.actions[:,-1] = torch.clip(actions[:,-1], self.cfg.normalization.clip_phase_actions_low, self.cfg.normalization.clip_phase_actions_high).to(self.device)

    def _generate_reference_motion(self):
        local_time = self.time % self.mocap_cycle_period
        local_time_plus_init = (local_time + self.init_mocap_data_idx*self.mocap_cycle_dt) % self.mocap_cycle_period
        self.mocap_data_idx = (self.init_mocap_data_idx + (local_time / self.mocap_cycle_dt).type(torch.long)) % self.mocap_data_num
        next_idx = self.mocap_data_idx + 1 

        mocap_data_idx_list = self.mocap_data_idx.squeeze(dim=-1)
        next_idx_list = next_idx.squeeze(dim=-1)

        self.target_data_qpos = cubic(local_time_plus_init, self.mocap_data[mocap_data_idx_list,0].unsqueeze(-1), self.mocap_data[next_idx_list,0].unsqueeze(-1), 
                                        self.mocap_data[mocap_data_idx_list,1:34], self.mocap_data[next_idx_list,1:34], 0.0, 0.0)
        self.target_data_force = cubic(local_time_plus_init, self.mocap_data[mocap_data_idx_list,0].unsqueeze(-1), self.mocap_data[next_idx_list,0].unsqueeze(-1), 
                                        self.mocap_data[mocap_data_idx_list,34:], self.mocap_data[next_idx_list,34:], 0.0, 0.0)

    def _process_dof_noise(self):
        self.qpos_noise = self.dof_pos + torch.clamp(torch.normal(torch.zeros_like(self.dof_pos), 0.00016/3.0), min=-0.00016, max=0.00016)
        self.qvel_noise = (self.qpos_noise - self.qpos_pre) / self.dt_sim
        self.qpos_pre = self.qpos_noise.clone()