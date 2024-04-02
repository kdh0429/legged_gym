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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import os
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

class TOCABIRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 128
        num_single_step_obs = 37
        num_obs_hist = 10
        num_obs_skip = 2
        num_actuator_action = 12
        num_actions = 13
        num_observations = (num_single_step_obs+num_actions)*(num_obs_hist-1)+num_single_step_obs
        episode_length_s = 32 # episode length in seconds
        
    class sim( LeggedRobotCfg.sim ):
        dt =  0.002

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class commands( LeggedRobotCfg.commands ):
        num_commands = 2 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 8. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.8, 0.8]   # min max [m/s]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.93] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "L_HipYaw_Joint": 0.0,
            "L_HipRoll_Joint": 0.0,
            "L_HipPitch_Joint": -0.24,
            "L_Knee_Joint": 0.6,
            "L_AnklePitch_Joint": -0.36,
            "L_AnkleRoll_Joint": 0.0,

            "R_HipYaw_Joint": 0.0,
            "R_HipRoll_Joint": 0.0,
            "R_HipPitch_Joint": -0.24,
            "R_Knee_Joint": 0.6,
            "R_AnklePitch_Joint": -0.36,
            "R_AnkleRoll_Joint": 0.0,

            "Waist1_Joint": 0.0,
            "Waist2_Joint": 0.0,
            "Upperbody_Joint": 0.0,

            "L_Shoulder1_Joint": 0.3,
            "L_Shoulder2_Joint": 0.3,
            "L_Shoulder3_Joint": 1.5,
            "L_Armlink_Joint": -1.27,
            "L_Elbow_Joint": -1.0,
            "L_Forearm_Joint": 0.0,
            "L_Wrist1_Joint": -1.0,
            "L_Wrist2_Joint": 0.0,

            "Neck_Joint": 0.0,
            "Head_Joint": 0.0,            
            
            "R_Shoulder1_Joint": -0.3,
            "R_Shoulder2_Joint": -0.3,
            "R_Shoulder3_Joint": -1.5,
            "R_Armlink_Joint": 1.27,
            "R_Elbow_Joint": 1.0,
            "R_Forearm_Joint": 0.0,
            "R_Wrist1_Joint": 1.0,
            "R_Wrist2_Joint": 0.0,
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'T' # P: position, V: velocity, T: torques

        p_gains = {"L_HipYaw_Joint": 2000.0, "L_HipRoll_Joint": 5000.0, "L_HipPitch_Joint": 4000.0,
                "L_Knee_Joint": 3700.0, "L_AnklePitch_Joint": 3200.0, "L_AnkleRoll_Joint": 3200.0,
                "R_HipYaw_Joint": 2000.0, "R_HipRoll_Joint": 5000.0, "R_HipPitch_Joint": 4000.0,
                "R_Knee_Joint": 3700.0, "R_AnklePitch_Joint": 3200.0, "R_AnkleRoll_Joint": 3200.0,

                "Waist1_Joint": 6000.0, "Waist2_Joint": 10000.0, "Upperbody_Joint": 10000.0,

                "L_Shoulder1_Joint": 400.0, "L_Shoulder2_Joint": 1000.0, "L_Shoulder3_Joint": 400.0, "L_Armlink_Joint": 400.0,
                "L_Elbow_Joint": 400.0, "L_Forearm_Joint": 400.0, "L_Wrist1_Joint": 100.0, "L_Wrist2_Joint": 100.0,

                "Neck_Joint": 100.0, "Head_Joint": 100.0,            

                "R_Shoulder1_Joint": 400.0, "R_Shoulder2_Joint": 1000.0, "R_Shoulder3_Joint": 400.0, "R_Armlink_Joint": 400.0,
                "R_Elbow_Joint": 400.0, "R_Forearm_Joint": 400.0, "R_Wrist1_Joint": 100.0, "R_Wrist2_Joint": 100.0}

        d_gains = {"L_HipYaw_Joint": 15.0, "L_HipRoll_Joint": 50.0, "L_HipPitch_Joint": 20.0,
                "L_Knee_Joint": 25.0, "L_AnklePitch_Joint": 24.0, "L_AnkleRoll_Joint": 24.0,
                "R_HipYaw_Joint": 15.0, "R_HipRoll_Joint": 50.0, "R_HipPitch_Joint": 20.0,
                "R_Knee_Joint": 25.0, "R_AnklePitch_Joint": 24.0, "R_AnkleRoll_Joint": 24.0,

                "Waist1_Joint": 200.0, "Waist2_Joint": 100.0, "Upperbody_Joint": 100.0,

                "L_Shoulder1_Joint": 10.0, "L_Shoulder2_Joint": 28.0, "L_Shoulder3_Joint": 10.0, "L_Armlink_Joint": 10.0,
                "L_Elbow_Joint": 10.0, "L_Forearm_Joint": 10.0, "L_Wrist1_Joint": 3.0, "L_Wrist2_Joint": 3.0,

                "Neck_Joint": 100.0, "Head_Joint": 100.0,            

                "R_Shoulder1_Joint": 10.0, "R_Shoulder2_Joint": 28.0, "R_Shoulder3_Joint": 10.0, "R_Armlink_Joint": 10.0,
                "R_Elbow_Joint": 10.0, "R_Forearm_Joint": 10.0, "R_Wrist1_Joint": 3.0, "R_Wrist2_Joint": 3.0}

        actuator_scale = [333, 232, 263, 289, 222, 166, \
                        333, 232, 263, 289, 222, 166, \
                        303, 303, 303, \
                        64, 64, 64, 64, 23, 23, 10, 10,\
                        10, 10, \
                        64, 64, 64, 64, 23, 23, 10, 10]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/tocabi/xml/dyros_tocabi.xml"
        name = "tocabi"
        foot_name = "Foot_Link"
        terminate_after_contacts_on = ["base_link", 
                                    "L_HipRoll_Link", "L_HipCenter_Link", "L_Thigh_Link", "L_Knee_Link", "L_AnkleCenter_Link", "L_AnkleRoll_Link", "L_Foot_Redundant_Link", \
                                    "R_HipRoll_Link", "R_HipCenter_Link", "R_Thigh_Link", "R_Knee_Link", "R_AnkleCenter_Link", "R_AnkleRoll_Link", "R_Foot_Redundant_Link", \
                                    "Waist1_Link", "Waist2_Link", "Upperbody_Link", \
                                    "L_Shoulder1_Link", "L_Shoulder2_Link", "L_Shoulder3_Link", "L_Armlink_Link", "L_Elbow_Link", "L_Forearm_Link", "L_Wrist1_Link", "L_Wrist2_Link", \
                                    "Neck_Link", "Head_Link", \
                                    "R_Shoulder1_Link", "R_Shoulder2_Link", "R_Shoulder3_Link", "R_Armlink_Link", "R_Elbow_Link", "R_Forearm_Link", "R_Wrist1_Link", "R_Wrist2_Link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            lin_vel_z = 0
            ang_vel_xy = 0
            orientation = 0
            torques = 0.1
            dof_vel = 0
            dof_acc = 0
            base_height = 0
            feet_air_time =  0
            collision = 0
            feet_stumble = 0
            action_rate = 0
            stand_still = 0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

    class normalization( LeggedRobotCfg.normalization ):
        clip_actuator_actions_high = 1.
        clip_actuator_actions_low = -1.
        clip_phase_actions_high = 1.
        clip_phase_actions_low = 0.
        phase_scale = 5

        obs_mean_dir = os.path.join(LEGGED_GYM_ENVS_DIR, 'tocabi/normalization/obs_mean_fixed.txt')
        obs_var_dir = os.path.join(LEGGED_GYM_ENVS_DIR, 'tocabi/normalization/obs_variance_fixed.txt')

    class noise( LeggedRobotCfg.noise ):
        add_noise = False

    class motion:
        file = os.path.join(LEGGED_GYM_ENVS_DIR, 'tocabi/motion/processed_data_tocabi_walk.txt')

class TOCABIRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1/20.0
        actor_hidden_dims = [256, 256]
        critic_hidden_dims = [256, 256]
        activation = 'relu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'rough_tocabi'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt