import isaacgym.gymapi as gymapi
import isaacgym.gymtorch as gymtorch
import isaacgym.gymutil as gymutil

import numpy as np
import os
import re
import sys
import torch
import time

import engines.engine as engine

class IsaacGymEngineSimplified(engine.Engine):
    def __init__(self, config, num_envs, device, visualize=False, control_mode=None):
        super().__init__()

        physics_engine = gymapi.SIM_PHYSX

        self._device = device
        self._num_envs = num_envs
        self._enable_viewer_sync = True
        self._asset_cache = dict()
        
        # Load the gym API to get access to the gym functions
        self._gym = gymapi.acquire_gym()

        sim_freq = config.get('sim_freq', 60)
        control_freq = config.get('control_freq', 10)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            f"Simulation frequency {sim_freq} must be greater than or equal to control frequency {control_freq} and an integer multiple of it."
        
        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)

        sim_timestep = 1.0 / sim_freq
        # Create the simulator with the given configuration and physics engine
        self._sim = self._create_simulator(physics_engine, config, sim_timestep, visualize)
        
        # Distance in meter between each envs
        self._env_spacing = config['env_spacing']
        self._envs = []
        
        # Set the control mode for the engine
        if (control_mode is not None):
            self._control_mode = control_mode
        elif 'control_mode' in config:
            self._control_mode = engine.ControlMode[config['control_mode']]
        else:
            self._control_mode = engine.ControlMode.none
        
        # Keep the list of kp, kd, and torque limits for each actors
        self._actor_kp = [[] for _ in range(self._num_envs)]
        self._actor_kd = [[] for _ in range(self._num_envs)]
        self._actor_torque_limits = [[] for _ in range(self._num_envs)]

        self._apply_forces_callback = None
        self._build_ground_plane()

        if visualize:
            self._build_viewer()
            self._prev_frame_time = 0.0
        
        return

    def creat_env(self):
        """ Create a new environment in the simulator, keep the environment pointer and return the id
        """

        env_spacing = self._get_env_spacing()
        num_envs = self.get_num_envs()
        num_env_per_row = int(np.sqrt(num_envs))
        lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        env_ptr = self._gym.creat_env(self._sim, lower, upper, num_env_per_row)

        env_id = len(self._envs)
        self._envs.append(env_ptr)  

        return env_id
    
    def finalize_sim(self):
        """ Finalize the simulator after all environments and actors have been created
        """

        self._gym.prepare_sim(self._sim)
        self._build_sim_tensors()

        return
    
    def step(self):
        """ Step the simulation forward by sim_steps 
        """
        # pushes commanded actuator targets into simulator tensors depending on control mode (pos/vel/etc).
        self._apply_cmd()

        for i in range(self._sim_steps):
            # 
            self._pre_sim_step()
            self._sim_step()
        
        self._refresh_sim_tensors()
        return
    
    def update_sim_state(self)"
        """ Update the simulation state from the simulator tensors
        """
        actor_ids = self._need_reset_buf.nonzero(as_tuple=False)
        actor_ids = actor_ids.type(torch.int32).flatten()

        if (len(actor_ids) > 0):
            # Set the states of the actors that need to be reset based on the ids
            self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                          gymtorch.unwrap_tensor(self._root_state_raw),
                                                          gymtorch.unwrap_tensor(actor_ids),
                                                          len(actor_ids))
            

        return
    
    def _build_sim_tensors(self):
        """ Build the simulation tensors from the simulator
        """
        # Get the raw state tensors from the simulator
        root_state_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        body_state_tensor = self._gym.acquire_rigid_body_state_tensor(self._sim)
        contact_force_tensor = self._gym.acquire_net_contact_force_tensor(self._sim)
        
        # Wrap the raw tensors with torch tensors for easier manipulation
        self._root_state_raw = gymtorch.wrap_tensor(root_state_tensor)
        self._dof_state_raw = gymtorch.wrap_tensor(dof_state_tensor)
        self._body_state_raw = gymtorch.wrap_tensor(body_state_tensor)
        self._contact_force_raw = gymtorch.wrap_tensor(contact_force_tensor)
        
        # Get the number of envs and actors
        num_envs = self.get_num_envs()
        num_actors = self._root_state_raw.shape[0]  # Should be 1 since we have only one quadruped actor per env
        self._actors_per_env = num_actors // num_envs

        # Reshape the root state tensors to num_env, actors_per_env, root_state_raw.shape

