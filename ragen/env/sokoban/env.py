import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
import json
import os
from .utils import (
    generate_room,
    collect_entity_coordinates,
    format_coordinate_render,
)
# from gym_sokoban.envs.sokoban_env.utils import generate_room
from ragen.env.base import BaseDiscreteActionEnv
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.utils import all_seed

class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):
    def __init__(self, config=None, **kwargs):
        self.config = config or SokobanEnvConfig()
        self.GRID_LOOKUP = self.config.grid_lookup
        self.ACTION_LOOKUP = self.config.action_lookup
        self.search_depth = self.config.search_depth
        self.max_solution_length = self.config.max_solution_length
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.render_mode = self.config.render_mode
        self.observation_format = self.config.observation_format

        self._datasets = {}  # lazy-loaded, keyed by mode ("train"/"val")

        BaseDiscreteActionEnv.__init__(self)
        GymSokobanEnv.__init__(
            self,
            dim_room=self.config.dim_room,
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
            **kwargs
        )

    def _get_dataset(self, mode):
        """Lazy-load train.parquet or val.parquet from dataset_dir."""
        if mode not in self._datasets:
            import datasets as hf_datasets
            path = os.path.join(self.config.dataset_dir, f"{mode}.parquet")
            self._datasets[mode] = hf_datasets.load_dataset("parquet", data_files=path)['train']
        return self._datasets[mode]

    def reset(self, seed=None, mode=None):
        if self.config.dataset_dir is not None:
            ds_mode = mode if mode in ("train", "val") else "train"
            ds = self._get_dataset(ds_mode)
            index = seed % len(ds) if seed is not None else 0
            data = ds[index]
            self.room_fixed = np.array(json.loads(data['room_fixed']))
            self.room_state = np.array(json.loads(data['room_state']))
            self.box_mapping = json.loads(data['box_mapping'])
            self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
            self.player_position = np.argwhere(self.room_state == 5)[0]
            return self.render()

        try:
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth,
                    max_solution_length=self.max_solution_length
                )
            self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
            self.player_position = np.argwhere(self.room_state == 5)[0]
            return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)
        
    def step(self, action: int):
        previous_pos = self.player_position
        _, reward, done, _ = GymSokobanEnv.step(self, action) 
        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)
        info = {"action_is_effective": action_effective, "action_is_valid": True, "success": self.boxes_on_target == self.num_boxes}
        return next_obs, reward, done, info

    def render(self, mode=None):
        if mode in {'grid', 'coord', 'grid_coord'}:
            return self._render_text(mode)

        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            return self._render_text(self.observation_format)
        if render_mode == 'rgb_array':
            return self.get_image(mode='rgb_array', scale=1)
        raise ValueError(f"Invalid mode: {render_mode}")

    def _render_text(self, observation_format: str) -> str:
        if observation_format == 'grid':
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        if observation_format == 'coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return format_coordinate_render(entity_coords, self.dim_room)
        if observation_format == 'grid_coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return "Coordinates: \n" + format_coordinate_render(entity_coords, self.dim_room) + "\n" + "Grid Map: \n" + self._render_text('grid')
        raise ValueError(f"Invalid observation_format: {observation_format}")
    
    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.keys()])
    
    def close(self):
        self.render_cache = None
        super(SokobanEnv, self).close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
    env = SokobanEnv(config)
    for i in range(10):
        print(env.reset(seed=1010 + i))
        print()
    while True:
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = int(keyboard)
        assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    np_img = env.get_image('rgb_array')
    # save the image
    plt.imsave('sokoban1.png', np_img)
