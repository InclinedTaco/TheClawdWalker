# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import numpy as np
import os
import torch
from torch import Tensor
from typing import Tuple, Dict
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from isaacgym import gymtorch
from isaacgym import gymapi


from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp , quat_conjugate
from isaacgymenvs.tasks.base.vec_task import VecTask

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaCubeStack(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.start_min_height_offset = 0.01
        self.start_max_height_offset = 0.30
        self.ik_control_damping = 0.25


        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": 10.0,  
            "r_lift_scale": 5.0,   
            "r_align_scale": 0.0,  
            "r_stack_scale": 0.0,  
            "r_action_rate_scale": 0.01,
            "r_torque_scale": 0.0001,
            "r_imitation_scale": 1.5
        }

        # Controller type
        self.control_type = "joint_tor"  # osc, joint_tor, visualize_ik_target
        assert self.control_type in {"osc", "joint_tor", "visualize_ik_target"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor, visualize_ik_target}"
        

        if self.control_type == "visualize_ik_target":
            self.plot_data = {"steps": [], "q_current": [], "q_target": []}
            self.max_plot_steps = 500  # Collect data for this many steps
            self.plot_generated = False  # Flag to ensure we only plot once

    
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 12 if self.control_type == "osc" else 19
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self.hold_time_achieved_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.success_timer_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_hold_steps = 60
        self.last_actions=torch.zeros(self.num_envs,self.num_actions,device=self.device)
        self.global_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.q_ref = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)


        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
         
        #BeyondMimic (for the arm)
        franka_dof_stiffness = to_torch([2368.7, 2368.7, 1776.5, 1776.5, 1776.5, 789.6, 789.6, 800.0, 800.0], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([150.8, 150.8, 113.1, 113.1, 113.1, 50.3, 50.3, 40.0, 40.0], dtype=torch.float, device=self.device)

        # franka_dof_stiffness= to_torch([400, 400, 400, 400, 400, 400, 400, 800.0, 800.0], dtype=torch.float, device=self.device)
        # franka_dof_damping = to_torch([40, 40, 40, 40, 40, 40, 40, 40.0, 40.0], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.cubeA_size = 0.02
        self.cubeB_size = 0.00000001

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_opts.fix_base_link = True
        cubeA_opts.disable_gravity = True
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cubeB asset
        cubeB_opts = gymapi.AssetOptions()
        cubeB_opts.fix_base_link = True
        cubeB_opts.disable_gravity = True
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS  #if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.robot_base_pos = to_torch([-0.5, 0.0], device=self.device)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self._table_surface_pos = to_torch(self._table_surface_pos, device=self.device)

        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4     # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 4     # 1 for table, table stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i+1, 0, 0)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
    

    def control_ik(self,dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self._j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.ik_control_damping ** 2)
        u = (j_eef_T @ torch.inverse(self._j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    # def orientation_error(self, desired, current):
    #     cc = quat_conjugate(current)
    #     q_r = quat_mul(desired, cc)
    #     return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            "cubeB_quat": self._cubeB_state[:, 3:7],
            "cubeB_pos": self._cubeB_state[:, :3],
            "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
            "last_actions": self.last_actions,
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        # Unpack all returned values, including the new metrics dictionary
        self.rew_buf[:], self.reset_buf[:], self.success_timer_buf[:], self.hold_time_achieved_buf[:], metrics_dict = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, 
            self.max_episode_length, self.success_timer_buf, self.success_hold_steps, self.q_ref
        )

        # Populate self.extras with the metrics for the logger
        # The isaacgymenvs runner will automatically log everything in this dictionary
        self.extras.update(metrics_dict)

    def compute_observations(self):
        self._refresh()
        # Simplified observations: just cube position and end effector state
        obs = ["cubeA_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes, sampling cube B first, then A
        # if not self._i:
        self._reset_init_cube_state(cube='B', env_ids=env_ids)
        self._reset_init_cube_state(cube='A', env_ids=env_ids)
        # self._i = True

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_cubes(self, env_ids):
        # Reset cubes, sampling cube B first, then A
        self._reset_init_cube_state(cube='B', env_ids=env_ids)
        self._reset_init_cube_state(cube='A', env_ids=env_ids)

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]
        
        # Update cube states in the simulation
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        
    def _reset_init_cube_state(self, cube, env_ids):
            """
            Simplified method to sample cube positions.
            - Cube B is set to a fixed, out-of-the-way location.
            - Cube A is randomized in a safe zone, away from the robot base.
            """
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            num_resets = len(env_ids)

            if cube.lower() == 'b':
                # --- Cube B: Set to a fixed position ---
                sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)
                
                # Position it at a fixed corner of the table, out of the way
                sampled_cube_state[:, 0] = 0.4
                sampled_cube_state[:, 1] = 0.4
                sampled_cube_state[:, 2] = self._table_surface_pos[2] + self.cubeB_size / 2
                sampled_cube_state[:, 6] = 1.0  # Default orientation
                
                self._init_cubeB_state[env_ids, :] = sampled_cube_state

            elif cube.lower() == 'a':
                # --- Cube A: Randomize in a safe zone ---
                sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)
                
                # Set fixed height and orientation
                base_z = self._table_surface_pos[2] + self.cubeA_size / 2
                # Add a random offset between min and max height settings
                height_range = self.start_max_height_offset - self.start_min_height_offset
                random_offsets = height_range * torch.rand(num_resets, device=self.device) + self.start_min_height_offset
                sampled_cube_state[:, 2] = base_z + random_offsets
                sampled_cube_state[:, 6] = 1.0

                # Loop to find a valid XY position
                active_idx = torch.arange(num_resets, device=self.device)
                for _ in range(100): # Max 100 attempts to find a valid spot
                    # Sample random XY positions for the active environments
                    rand_xy = 2.0 * self.start_position_noise * (torch.rand(len(active_idx), 2, device=self.device) - 0.5)
                    sampled_cube_state[active_idx, :2] = self._table_surface_pos[:2] + rand_xy

                    # Check if the sampled points are inside the robot's no-spawn zone
                    dist_from_robot_base = torch.linalg.norm(sampled_cube_state[active_idx, :2] - self.robot_base_pos, dim=-1)
                    invalid_points = dist_from_robot_base < 0.30 #self.robot_base_clearance
                    
                    # If all points are valid, we're done
                    if not torch.any(invalid_points):
                        break
                    
                    # Update the active indices to only include the invalid ones that need re-sampling
                    active_idx = active_idx[invalid_points]
                
                self._init_cubeA_state[env_ids, :] = sampled_cube_state
                
    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u
    
    def plot_results(self):
        # Check if any data was collected
        if not self.plot_data["steps"]:
            print("No data collected for plotting.")
            return

        print(f"--- Generating joint tracking plot from {len(self.plot_data['steps'])} data points ---")
        
        q_current = np.array(self.plot_data["q_current"])
        q_target = np.array(self.plot_data["q_target"])
        steps = np.arange(len(self.plot_data["steps"]))

        fig, axs = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
        fig.suptitle('Franka Joint Position Tracking (IK Target vs. Actual)', fontsize=16)

        for i in range(7):
            axs[i].plot(steps, q_current[:, i], 'b-', label='Current Joint Position')
            axs[i].plot(steps, q_target[:, i], 'r--', label='IK Target Position')
            axs[i].set_ylabel(f'Joint {i} (rad)')
            axs[i].grid(True)
            axs[i].legend()

        axs[-1].set_xlabel('Time Step')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_filename = "joint_tracking_plot.png"
        plt.savefig(plot_filename)
        plt.close() # Frees up memory
        print(f"--- Plot saved to {plot_filename} ---")

    def pre_physics_step(self, actions):
            self.actions = actions.clone().to(self.device)

            # --- Common IK Calculation ---
            goal_pos = self.states["cubeA_pos"]
            hand_pos = self.states["eef_pos"]
            hand_rot = self.states["eef_quat"]

            # Define a target orientation: pointing straight down (180-degree rotation around the x-axis)
            # (w, x, y, z) format for gymapi.Quat, but torch_jit_utils uses (x, y, z, w)
            down_q = to_torch([1.0, 0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))

            # Calculate position and orientation error
            pos_error = goal_pos - hand_pos
            orn_error = orientation_error(down_q, hand_rot)

            dpose = torch.cat((pos_error, orn_error), dim=-1).unsqueeze(-1)

            delta_q = self.control_ik(dpose)
            
            # This is the IK target we want to visualize
            self.q_ref = self._q[:, :7] + delta_q

            # --- Control Logic Branching ---

            if self.control_type == "retarget":
                '''
                KINEMATIC TELEPORT MODE
                This block bypasses physics and directly sets the joint angles to the IK target.
                '''
                # 1. Get a writable copy of the full DOF state tensor
                dof_state_copy = self._dof_state.clone()

                # 2. Set the arm joint positions (DOFs 0-6) to the IK target
                dof_state_copy[:, :7, 0] = self.q_ref

                # 3. Set the gripper joint positions (DOFs 7-8) based on the action
                u_gripper = self.actions[:, -1]
                gripper_target_pos = torch.where(u_gripper.unsqueeze(-1) >= 0.0,
                                                self.franka_dof_upper_limits[-2:],
                                                self.franka_dof_upper_limits[-2:])
                dof_state_copy[:, 7:9, 0] = gripper_target_pos
                
                # 4. Zero out all joint velocities for a clean "snap"
                dof_state_copy[:, :, 1] = 0.0

                # 5. Apply the modified DOF state directly to the simulation
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state_copy))

                # 6. Skip the physics-based commands at the end of the function
                return
            
            elif self.control_type == "visualize_ik_target":
                '''
                BUILT-IN SIMULATOR POSITION CONTROL
                This block sets the target joint angles and lets the simulator's
                internal PID controller handle the physics.
                '''
                # --- Arm Control ---
                # 1. Set the arm's target angles directly to the IK solution
                self._pos_control[:, :7] = self.q_ref

                # --- Gripper Control ---
                # 2. Set the gripper's target angle based on the action
                u_gripper = self.actions[:, -1]
                gripper_target_pos = torch.where(u_gripper.unsqueeze(-1) >= 0.0,
                                                 self.franka_dof_upper_limits[-2:],
                                                 self.franka_dof_lower_limits[-2:])
                # Set the gripper's target position in the buffer
                self._pos_control[:, 7:9] = self.franka_dof_upper_limits[-2:]
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

                
            elif self.control_type == "joint_tor":
                '''
                ORIGINAL CODE FOR TRAINING/RUNNING WITH TORQUE CONTROL
                '''
                beta = 100 * delta_q - 20 * self._qd[:, :7]
                t = self.global_step_buf
                decay_term = 0.99**(t / 100)
                u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]
                u_arm = u_arm * self.cmd_limit / self.action_scale
                final_torques = u_arm + decay_term.unsqueeze(-1) * beta

                if self.control_type == "osc":
                    print("OSC not yet")
                    # u_arm = self._compute_osc_torques(dpose=u_arm)
                
                final_torques = tensor_clamp(final_torques,
                                            -self._franka_effort_limits[:7],
                                            self._franka_effort_limits[:7])
                self._arm_control[:, :] = final_torques

                # Control gripper
                u_fingers = torch.zeros_like(self._gripper_control)
                u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                            self.franka_dof_lower_limits[-2].item())
                u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                            self.franka_dof_lower_limits[-1].item())
                self._gripper_control[:, :] = u_fingers

                # Deploy actions
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

            # Update last actions buffer
            # self.last_actions[:] = self.actions[:]

    def post_physics_step(self):
        self.progress_buf += 1
        self.global_step_buf += 1

        if self.control_type == "visualize_ik_target" and not self.plot_generated:
            # Check if we still need to collect more data
            if len(self.plot_data["steps"]) < self.max_plot_steps:
                # Store data from the first environment
                self.plot_data["steps"].append(self.global_step_buf[0].item())
                self.plot_data["q_current"].append(self._q[0, :7].cpu().numpy())
                self.plot_data["q_target"].append(self.q_ref[0, :7].cpu().numpy())
            else:
                # We have enough data, so generate the plot and set the flag
                self.plot_results()
                self.plot_generated = True

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.global_step_buf[env_ids] = 0

        cube_reset_env_ids=self.hold_time_achieved_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(cube_reset_env_ids) > 0:
            self._reset_cubes(cube_reset_env_ids)
            self.success_timer_buf[cube_reset_env_ids]=0
            self.progress_buf[cube_reset_env_ids]=0

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

        self.last_actions[:] = self.actions[:]

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf: Tensor, 
    progress_buf: Tensor, 
    actions: Tensor, 
    states: Dict[str, Tensor], 
    reward_settings: Dict[str, float], 
    max_episode_length: float, 
    success_timer_buf: Tensor, 
    success_hold_steps: int, 
    q_ref: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:

    # --- Calculate Individual Reward Components ---
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    dist_reward = reward_settings["r_dist_scale"] * (1 - torch.tanh(10.0 * d))
    close_reward = reward_settings["r_lift_scale"] * (d < 0.05).float()

    joint_error = torch.sum(torch.square(q_ref - states["q"][:, :7]), dim=-1)
    sigma_imitation = 0.1
    imitation_reward = reward_settings["r_imitation_scale"] * torch.exp(-joint_error / sigma_imitation)
    
    if torch.sum(torch.square(actions)) == 0.0:
        print("DEBUG: Current actions are all zero.")
    if torch.sum(torch.square(states["last_actions"])) == 0.0:
        print("DEBUG: Last actions are all zero.")

    action_rate_penalty = -reward_settings["r_action_rate_scale"] * torch.norm(actions - states["last_actions"], dim=-1)
    torque_penalty = -reward_settings["r_torque_scale"] * torch.sum(torch.square(actions), dim=-1)

    # --- Final Combined Reward ---
    rewards = dist_reward + close_reward + imitation_reward + action_rate_penalty + torque_penalty

    # --- METRIC CALCULATIONS (e.g., RMSE) ---
    joint_pos_error_sq = torch.square(q_ref - states["q"][:, :7])
    joint_pos_rmse = torch.sqrt(torch.mean(joint_pos_error_sq))

    # --- Create Dictionary for Logging ---
    # We log the mean of per-environment values and single-value metrics
    log_metrics = {
        "rewards/dist_reward": torch.mean(dist_reward),
        "rewards/close_reward": torch.mean(close_reward),
        "rewards/imitation_reward": torch.mean(imitation_reward),
        "rewards/action_rate_penalty": torch.mean(action_rate_penalty),
        "rewards/torque_penalty": torch.mean(torque_penalty),
        "rmse/joint_pos_imitation": joint_pos_rmse,
    }

    # --- Episode End Logic ---
    # Use a stricter threshold for success condition if desired
    close_to_cube_for_hold = d < 0.045 
    success_timer_buf = torch.where(close_to_cube_for_hold, success_timer_buf + 1, torch.zeros_like(success_timer_buf))
    hold_time_achieved = success_timer_buf >= success_hold_steps
    
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf, success_timer_buf, hold_time_achieved, log_metrics
