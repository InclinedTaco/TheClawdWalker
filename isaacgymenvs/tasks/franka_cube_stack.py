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
import pytorch_kinematics as pk
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
from isaacgym import gymutil


from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp , quat_conjugate, quat_apply
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

        self.max_episode_length = 500 #self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.go2_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.start_min_height_offset = 0.01
        self.start_max_height_offset = 0.30
        self.ik_control_damping = 0.25


        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": 10.0,  
            "r_lift_scale": 10.0,   
            "r_align_scale": 0.0,  
            "r_stack_scale": 0.0,  
            "r_action_rate_scale": 0.01,
            "r_torque_scale": 0.0001,
            "r_imitation_scale": 1.5
        }

        # Controller type
        self.control_type = "ik_position" 
        assert self.control_type in {"ik_pd","rl","ik_position"},\
            "Invalid control type specified. Must be one of: {ik_pd , rl, ik_position}"


        if self.control_type == "visualize_ik_target":
            self.plot_data = {"steps": [], "q_current": [], "q_target": []}
            self.max_plot_steps = 500  # Collect data for this many steps
            self.plot_generated = False  # Flag to ensure we only plot once


        self.cfg["env"]["numObservations"] = 40

        self.cfg["env"]["numActions"] = 18

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env


        self._cubeA_state = None                # Current state of cubeA for the current env

        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env

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

        self.action_plot_generated = False
        self.action_data_plotting = {
                    "steps": [],
                    "action_prior": [],
                    "policy_action": [],
                    "total_action": [],
                    "decay_factor": [],
                    "ik_target": [],
                    "smooth_ik_target": [],
                    "pd_torques": [],
                    "q_current": [],
                    "beta": []
                }

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.asset_root = "/home/marmot/Shyam/go2_with_airbot"
        self.go2_asset_file = "urdf/go2_with_airbot_delete_collision.urdf"
        urdf_path = os.path.join(self.asset_root, self.go2_asset_file)
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read())
        self.chain = self.chain.to(device=self.device) # Move the model to the GPU
        self.eef_link_name = "eef_end_link"
                
        # Get DOF names from PyTorch Kinematics
        pk_dof_names = self.chain.get_joint_parameter_names()
        print("\n--- PyTorch Kinematics DOF Order ---")
        for i, name in enumerate(pk_dof_names):
            print(f"{i}: {name}")

        

        self.hold_time_achieved_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.success_timer_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_hold_steps = 60
        self.last_actions=torch.zeros(self.num_envs,self.num_actions,device=self.device)
        self.global_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.q_ref = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)


        # Legs (angles for hip, thigh, calf):
        leg_default_pos = [0.0, 0.9, -1.8] * 4  # FL, FR, RL, RR
        # Arm (all joints at 0):
        arm_default_pos = [0.0] * 6
        self.go2_default_dof_pos = to_torch(leg_default_pos + arm_default_pos, device=self.device)

        # Set control limits
        self.cmd_limit = self.effort_limits[self.arm_dof_indices].unsqueeze(0)
        print(f"Control type: {self.control_type}")
        print(f"Command limits set to: {self.cmd_limit}")
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

    def clear_lines(self):
        if self.viewer:
            self.gym.clear_lines(self.viewer)

    def draw_sphere(self, pos, radius, color, env_id):
        if self.viewer:
            # Convert position to Vec3
            sphere_pos = gymapi.Vec3(pos[0], pos[1], pos[2])
            # Create a transform for the sphere
            sphere_pose = gymapi.Transform(p=sphere_pos, r=None)
            # Create the sphere geometry
            sphere_geom = gymutil.WireframeSphereGeometry(radius, 20, 20, None, color=color)
            # Draw the sphere
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.asset_root = "/home/marmot/Shyam/go2_with_airbot"
        self.go2_asset_file = "urdf/go2_with_airbot_delete_collision.urdf"



        # load  asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        

        go2_asset = self.gym.load_asset(self.sim, self.asset_root, self.go2_asset_file, asset_options)
        
        print("\n--- RIGID BODY INDICES FOR GO2 ASSET ---")
        body_names = self.gym.get_asset_rigid_body_names(go2_asset)
        for i, name in enumerate(body_names):
            print(f"Body Index {i}: {name}")
        print("----------------------------------------\n")


        isaac_dof_names = self.gym.get_asset_dof_names(go2_asset)
        print("--- Isaac Gym DOF Order ---")
        for i, name in enumerate(isaac_dof_names):
            print(f"{i}: {name}")

        
        self.num_dofs = self.gym.get_asset_dof_count(go2_asset)
        print("num dofs: ", self.num_dofs)
        self.num_bodies  = self.gym.get_asset_rigid_body_count(go2_asset)
        
        dof_props = self.gym.get_asset_dof_properties(go2_asset)
        dof_names = self.gym.get_asset_dof_names(go2_asset)

        #PD params
        leg_stiffness = 30.0
        leg_damping = 1.0
        self.arm_stiffness = to_torch([400.0, 400.0, 200.0, 200.0, 10.0, 10.0], device=self.device)
        self.arm_damping = to_torch([40.0, 40.0, 20.0, 20.0, 0.5, 0.5], device=self.device)

        # #BeyondMimic (for the arm)
        # # self.franka_dof_stiffness = to_torch([2368.7, 2368.7, 1776.5, 1776.5, 1776.5, 789.6, 789.6, 800.0, 800.0], dtype=torch.float, device=self.device)
        # # self.franka_dof_damping = to_torch([30.0, 30.0, 22.0, 22.0, 22.0, 10.0, 10.0, 40.0, 40.0], dtype=torch.float, device=self.device)

        self.leg_dof_indices = []
        self.arm_dof_indices = []
        arm_joint_counter = 0
        
        for i in range(self.num_dofs):
            name = dof_names[i]

            if "_hip" in name or "_thigh" in name or "_calf" in name:
                dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][i] = leg_stiffness
                dof_props['damping'][i] = leg_damping
                self.leg_dof_indices.append(i)

            elif "airbot_j" in name:
                dof_props["driveMode"][i] = gymapi.DOF_MODE_POS if self.control_type == "ik_position" else gymapi.DOF_MODE_EFFORT
                dof_props['stiffness'][i] = self.arm_stiffness[arm_joint_counter]
                dof_props['damping'][i] = self.arm_damping[arm_joint_counter]
                self.arm_dof_indices.append(i)
                arm_joint_counter += 1

        self.leg_dof_indices = to_torch(self.leg_dof_indices, device=self.device, dtype=torch.long)
        self.arm_dof_indices = to_torch(self.arm_dof_indices, device=self.device, dtype=torch.long)

        self.dof_lower_limits = to_torch(dof_props['lower'], device=self.device)
        self.dof_upper_limits = to_torch(dof_props['upper'], device=self.device)
        self.effort_limits = to_torch(dof_props['effort'], device=self.device)

        # Define start pose for franka
        go2_start_pose = gymapi.Transform()
        go2_start_pose.p = gymapi.Vec3(0.0,0.0,0.35)
        go2_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.robot_base_pos = to_torch([-0.5, 0.0], device=self.device)

        self.robots = []
        self.envs = []

        self.cubeA_size = 0.02

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_opts.fix_base_link = True
        cubeA_opts.disable_gravity = False
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)


        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            robot_actor = self.gym.create_actor(env_ptr, go2_asset, go2_start_pose, "go2_arm", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, dof_props)

            cubeA_start_pose = gymapi.Transform()
            cubeA_start_pose.p= gymapi.Vec3(10.0, 10.0, 10.0)

            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i+1, 0, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)


            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        env_ptr = self.envs[0]
        robot_handle = self.robots[0] 


        eef_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "eef_end_link")


        self.handles = {
            "trunk": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "trunk"),
            "eef": eef_handle,
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
        }

        print("\n--- RIGID BODY HANDLE VERIFICATION ---")
        print(f"Actor Handle for Robot: {robot_handle}")
        print(f"Searching for link name: 'eef_end_link'")
        print(f"Found EEF Handle: {self.handles['eef']} (Should be 40)")
        if self.handles['eef'] != 40:
            print("!!! WARNING: INCORRECT EEF HANDLE FOUND. IK WILL FAIL. !!!")

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "go2_arm")


        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._jacobian = gymtorch.wrap_tensor(_jacobian_tensor)
        

        self._root_state = self._root_state.view(self.num_envs, -1, 13) 
        
        # State of the robot's base (the first actor)
        self._base_state = self._root_state[:, 0, :]
        
        # State of the cube (the second actor)
        self._cubeA_state = self._root_state[:, 1, :]

        self._dof_state = self._dof_state.view(self.num_envs, self.num_dofs, 2)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        self._global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32,
                                        device=self.device).view(self.num_envs, -1)
        

        self.joint_4_raw_ik_output = []
        self.joint_4_smoothed_ik_output = []


    def control_ik(self,dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self._j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.ik_control_damping ** 2)
        u = (j_eef_T @ torch.inverse(self._j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
        return u
    
    def compute_fk(self, q):
        """
        Computes FK  for a batch of joint configs.
        """
        all_link_transforms = self.chain.forward_kinematics(q)
        eef_transforms = all_link_transforms[self.eef_link_name]
        eef_matrix = eef_transforms.get_matrix()
        eef_pos = eef_matrix[:, :3, 3]
        return eef_pos


    def _update_states(self):
            """
            Gathers all relevant states from the simulation tensors and stores them
            in a convenient dictionary, self.states.
            This is called once per simulation step in the _refresh() method.
            """

            eef_body_index = self.handles["eef"]
            self._j_eef = self._jacobian[:, eef_body_index, :, self.arm_dof_indices]
            relative_cube_pos = self._cubeA_state[:, :3] - self._rigid_body_state[:, self.handles["eef"], :3]

            self.states.update({

                "q_arm": self._q[:, self.arm_dof_indices],

                "qd_arm": self._qd[:, self.arm_dof_indices],

                "eef_pos": self._rigid_body_state[:, self.handles["eef"], :3],

                "eef_quat": self._rigid_body_state[:, self.handles["eef"], 3:7],

                "cubeA_pos": self._cubeA_state[:, :3],

                "cubeA_pos_relative": relative_cube_pos,

                "last_actions": self.last_actions
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
        """
        This method gathers the necessary state tensors from the environment
        and calls the JIT-compiled reward function.
        """

        
        # Calculate distance to cube
        dist_to_cube = torch.norm(self.states["cubeA_pos_relative"], dim=-1)
        
        # Get base state information
        base_lin_vel = self._base_state[:, 7:10]
        base_ang_vel = self._base_state[:, 10:13]
        trunk_height = self._base_state[:, 2]
        

        self.rew_buf[:], self.reset_buf[:], metrics = compute_go2_arm_reward(
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            base_lin_vel,
            base_ang_vel,
            trunk_height,
            dist_to_cube,
            self.max_episode_length
        )
        
        self.extras.update(metrics)

    def compute_observations(self):
        self._refresh()

        obs = ["cubeA_pos", "eef_pos", "eef_quat"]
        obs += ["q_arm", "qd_arm", "last_actions"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self._reset_init_cube_state(cube='A', env_ids=env_ids)
            self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]


            reset_noise = torch.rand((len(env_ids), self.num_dofs), device=self.device) 
            
            pos = tensor_clamp(
                self.go2_default_dof_pos.unsqueeze(0) +
                self.go2_dof_noise * 2.0 * (reset_noise - 0.5), 
                self.dof_lower_limits.unsqueeze(0), self.dof_upper_limits.unsqueeze(0))

            self._q[env_ids, :] = pos
            self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
            self._pos_control[env_ids, :] = pos
            self._effort_control[env_ids, :] = torch.zeros_like(pos)
            
            multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self._dof_state),
                                                gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                len(multi_env_ids_int32))

            multi_env_ids_cubes_int32 = self._global_indices[env_ids, 1].flatten()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self.last_actions[env_ids] = 0

    def _reset_cubes(self, env_ids):

        self._reset_init_cube_state(cube='A', env_ids=env_ids)

        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]

        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        
    def _reset_init_cube_state(self, cube, env_ids):
            """
            Resets the cube to a random position within the arm's workspace,
            relative to the robot's base.
            """
            if env_ids is None or len(env_ids) == 0:
                return

            num_resets = len(env_ids)

            robot_base_pos = self._base_state[env_ids, 0:3]

            min_fwd_dist = 0.33
            max_fwd_dist = 0.4

            max_side_dist = 0.20

            min_height_offset = 0.10
            max_height_offset = 0.45

            forward_offset = (min_fwd_dist + (max_fwd_dist - min_fwd_dist) * torch.rand(num_resets, device=self.device))

            side_offset = max_side_dist * (2.0 * torch.rand(num_resets, device=self.device) - 1.0)

            height_offset = (min_height_offset + (max_height_offset - min_height_offset) * torch.rand(num_resets, device=self.device))

            sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)
            

            sampled_cube_state[:, 0] = robot_base_pos[:, 0] + forward_offset
            sampled_cube_state[:, 1] = robot_base_pos[:, 1] + side_offset
            sampled_cube_state[:, 2] = robot_base_pos[:, 2] + height_offset
            

            cube_center_min_height = self.cubeA_size / 2.0
            sampled_cube_state[:, 2] = torch.clamp(sampled_cube_state[:, 2], min=cube_center_min_height)

            sampled_cube_state[:, 6] = 1.0

            if cube == 'A':
                self._init_cubeA_state[env_ids, :] = sampled_cube_state
                    
    
    def plot_results(self):

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
        plt.close() 
        print(f"--- Plot saved to {plot_filename} ---")

    def pre_physics_step(self, actions):
            self.actions = actions.clone().to(self.device)
            self.actions = self.actions * self.action_scale

            leg_target_pos = self.go2_default_dof_pos[self.leg_dof_indices]
            self._pos_control[:, self.leg_dof_indices] = leg_target_pos

            eef_pos = self.states["eef_pos"]
            eef_quat = self.states["eef_quat"]
            cube_pos = self.states["cubeA_pos"]
            q_current_arm = self.states["q_arm"]
            qd_current_arm = self.states["qd_arm"]


            down_q = to_torch([0, 0.7071, 0.0, 0.7071], device=self.device).repeat((self.num_envs, 1))
            pos_error = cube_pos - eef_pos
            # orn_error = torch.zeros_like(pos_error)
            orn_error = orientation_error(down_q, eef_quat)

            dpose = torch.cat((pos_error, orn_error), dim=-1).unsqueeze(-1)

  
            # if self.progress_buf[0] < 20: # Print for the first couple of steps
            #     print(f"\n--- Sliced Jacobian Check (Step {self.progress_buf[0].item()}) ---")
            #     print(self._j_eef[0].cpu().numpy().round(3))
            #     print("---------------------------------")

            delta_q = self.control_ik(dpose)
            
            self.q_ref = q_current_arm + delta_q
            self.alpha = 0.3
            self.smooth_q_target = torch.zeros_like(self.q_ref) if not hasattr(self, 'smooth_q_target') else self.smooth_q_target
            self.smooth_q_target = self.alpha * self.q_ref + (1.0 - self.alpha) * self.smooth_q_target
            self.pos_error_pd = self.smooth_q_target - q_current_arm
            self.vel_error_pd = -qd_current_arm  
            self.pd_torques = self.arm_stiffness * self.pos_error_pd + self.arm_damping * self.vel_error_pd

            self.decay_factor = 0.99
            self.t = self.global_step_buf.item()
            self.prior_weight = self.decay_factor ** (self.t / 100)
            self.action_prior = self.prior_weight * self.pd_torques

            self.final_arm_torques = torch.zeros_like(q_current_arm)

            if self.control_type == "ik_position":
                self._pos_control[:, self.arm_dof_indices] = self.smooth_q_target
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))


            elif self.control_type == "ik_pd":
                if self.progress_buf[0] < 5:
                    print(f"Cube Pos: {self.states['cubeA_pos'][0].cpu().numpy().round(3)},  EEF Pos: {self.states['eef_pos'][0].cpu().numpy().round(3)}")
                

                arm_effort_limits = self.effort_limits[self.arm_dof_indices]
                final_arm_torques = torch.clamp(self.pd_torques, -arm_effort_limits, arm_effort_limits)

                # Apply the final torques to the arm's effort control buffer
                self._effort_control[:, self.arm_dof_indices] = final_arm_torques


                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

            elif self.control_type == "rl":
                policy_pos = self.actions + self.go2_default_dof_pos[self.arm_dof_indices]
                policy_torque = self.arm_stiffness * (policy_pos - q_current_arm) - self.arm_damping * qd_current_arm

                total_torque = policy_torque + self.action_prior

                final_arm_torques = torch.clamp(total_torque, -arm_effort_limits, arm_effort_limits)

                self._effort_control[:, self.arm_dof_indices] = final_arm_torques
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

                

    def post_physics_step(self):
        self.progress_buf += 1
        self.global_step_buf += 1

        if self.viewer:
            self.clear_lines()

            q_ref_padded = self.go2_default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)

            q_ref_padded[:, self.arm_dof_indices] = self.smooth_q_target

            eef_pos_target_local = self.compute_fk(q_ref_padded)

            robot_base_pos = self._base_state[:, 0:3]
            robot_base_quat = self._base_state[:, 3:7]
            eef_pos_target_world = quat_apply(robot_base_quat, eef_pos_target_local) + robot_base_pos

            current_eef_world = self.states["eef_pos"]

            i = 55
            # Green sphere for the IK target
            self.draw_sphere(pos=eef_pos_target_world[i], radius=0.025, color=(0.1, 1.0, 0.1), env_id=i)
            # Red sphere for the robot's current EEF position
            self.draw_sphere(pos=current_eef_world[i], radius=0.02, color=(1.0, 0.1, 0.1), env_id=i)

        dist_to_cube = torch.norm(self.states["cubeA_pos_relative"], dim=-1)

        is_successful_now = (dist_to_cube < 0.05)

        self.success_buf = (self.success_buf + 1) * is_successful_now

        task_complete_env_ids = (self.success_buf >= 10).nonzero(as_tuple=False).squeeze(-1)

        if len(task_complete_env_ids) > 0:
            print(f"--- SUCCESS! Envs {task_complete_env_ids.cpu().numpy()} completed the task. Resetting cube. ---")
            self._reset_cubes(task_complete_env_ids)
            self.success_buf[task_complete_env_ids] = 0

            self.rew_buf[task_complete_env_ids] += 25.0

            self.rew_buf[task_complete_env_ids] += 25.0

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            # self.global_step_buf[env_ids] = 0

        cube_reset_env_ids=self.hold_time_achieved_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(cube_reset_env_ids) > 0:
            self._reset_cubes(cube_reset_env_ids)
            self.success_timer_buf[cube_reset_env_ids]=0
            self.progress_buf[cube_reset_env_ids]=0

        self.compute_observations()
        self.compute_reward(self.actions)

        self.last_actions[:] = self.actions[:]


    def plot_action_priors(self):
        """Generates and saves a plot of the decay factor and action components."""
        if not self.action_data_plotting["steps"]:
            print("No action prior data collected for plotting.")
            return

        print(f"--- Generating action prior plot from {len(self.action_data_plotting['steps'])} data points ---")

        steps = self.action_data_plotting["steps"]

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('Analysis of Decaying Action Priors (DecAP)', fontsize=16)

        # Subplot 1: IK Target
        axs[0].plot(steps, self.action_data_plotting["action_prior"], 'b', label='action_prior')
        axs[0].set_ylabel('Torque (Nm)')
        axs[0].set_title('Action Prior (β_t) Over Episode')
        axs[0].grid(True)
        axs[0].legend()
        data_min = np.min(self.action_data_plotting["action_prior"])
        data_max = np.max(self.action_data_plotting["action_prior"])
        padding = (data_max - data_min) * 0.1 # Add 10% padding
        axs[0].set_ylim(data_min - padding, data_max + padding)

        # Subplot 2: Torque Components for Joint 0
        # axs[1].plot(steps, self.action_data_plotting["action_prior"], 'r--', label='Action Prior')
        axs[1].plot(steps, self.action_data_plotting["beta"], 'b:', label='Non-decayed Torque')
        # axs[1].plot(steps, self.action_data_plotting["total_action"], 'k-', label='Final Torque (sent to robot)', linewidth=0.5)
        axs[1].set_ylabel('Torque (Nm)')
        axs[1].set_title('Non-decayed Torque Over Episode')
        axs[1].grid(True)
        axs[1].legend()
        data_min = np.min(self.action_data_plotting["beta"])
        data_max = np.max(self.action_data_plotting["beta"])
        padding = (data_max - data_min) * 0.1 # Add 10% padding
        axs[1].set_ylim(data_min - padding, data_max + padding)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_filename = "action_priors_plot.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"--- Action prior plot saved to {plot_filename} ---")

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_go2_arm_reward(
    # --- Inputs ---
    rew_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    actions: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    trunk_height: torch.Tensor,
    dist_to_cube: torch.Tensor,
    max_episode_length: float
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    # === 1. Calculate Reward Components ===
    
    # a) Reach Reward: Primary reward for getting close to the cube
    reach_reward = 1.0 * torch.exp(-10.0 * dist_to_cube)
    
    # b) Stationary Penalty: Penalize base movement
    stationary_penalty = torch.sum(torch.square(base_lin_vel), dim=-1) + \
                         torch.sum(torch.square(base_ang_vel), dim=-1)
    
    # c) Action Penalty: Encourage energy efficiency
    action_penalty = torch.sum(torch.square(actions), dim=-1)

    # === 2. Combine Rewards and Penalties (Weighted Sum) ===
    total_reward = reach_reward - 0.1 * stationary_penalty - 0.001 * action_penalty
    
    # === 3. Update Reset Buffer ===
    # Reset if fallen over, target is too far, or episode times out
    resets = torch.where(trunk_height < 0.2, 1, 0)
    resets = torch.where(dist_to_cube > 1.5, 1, resets)
    resets = torch.where(progress_buf >= max_episode_length - 1, 1, resets)
    
    # === 4. Logging Dictionary ===
    log_metrics = {
        "rewards/reach_reward": torch.mean(reach_reward),
        "rewards/total_reward": torch.mean(total_reward),
        "penalties/stationary_penalty": torch.mean(stationary_penalty),
        "penalties/action_penalty": torch.mean(action_penalty),
        "distance/eef_to_cube": torch.mean(dist_to_cube),
        "info/trunk_height": torch.mean(trunk_height)
    }

    return total_reward, resets, log_metrics
