import torch
import numpy as np
import os
import sys
import importlib
import json
import open3d as o3d

from utils import obj_surface_to_particles, obj_volume_to_particles, numpy_from_bgeo, write_particles
from default_torch import MyParticleNetwork


class Worker: 
    def __init__(self, weights_path, scene_name, output_dir, 
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        self.device = device
        # print("load scene")
        with open(scene_name, 'r') as f:
            scene = json.load(f)
        # print("makedir output_dir")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # print("create model")
        self.model = self.create_model()
        # print("get weights")
        weights = torch.load(weights_path)
        # print("load weights")
        self.model.load_state_dict(weights)
        # print("to device")
        self.model.to(self.device)
        self.model.requires_grad_(False)
        self.fluids = []
        self.walls = []
        self.scene = scene
        self.output_dir = output_dir
        # print("get walls and fluids particles")
        # self.walls, self.fluids = self.get_walls_and_fluids()
        self.wall = self.get_walls()
        self.fluids = self.get_fluids()
        # print("Init done!")
    
    def create_model(self):
        # print("In create_model")
        from default_torch import MyParticleNetwork
        # print("import MyParticleNetwork")
        """Returns an instance of the network for training and evaluation"""
        model = MyParticleNetwork()
        # print("get the model")
        return model


    def get_walls(self, ):
        walls = []
        for x in self.scene['walls']:
            points, normals = obj_surface_to_particles(x['path'])
            if 'invert_normals' in x and x['invert_normals']:
                normals = -normals
            points += np.asarray([x['translation']], dtype=np.float32)
            walls.append((points, normals))
        self.box = np.concatenate([x[0] for x in walls], axis=0)
        self.box_normals = np.concatenate([x[1] for x in walls], axis=0)
        
        self.min_y = np.min(self.box[:, 1]) - 0.05 * (np.max(self.box[:, 1]) - np.min(self.box[:, 1]))

        self.box = torch.from_numpy(self.box).to(self.device)   
        self.box_normals = torch.from_numpy(self.box_normals).to(self.device)
        return walls


    def get_fluids(self, ):
        fluids = []
        for x in self.scene['fluids']:
            points = obj_volume_to_particles(x['path'])[0]
            points += np.asarray([x['translation']], dtype=np.float32)
            velocities = np.empty_like(points)
            velocities[:, 0] = x['velocity'][0]
            velocities[:, 1] = x['velocity'][1]
            velocities[:, 2] = x['velocity'][2]
            range_ = range(x['start'], x['stop'], x['step'])
            fluids.append([points, velocities, range_])

        return fluids

    def get_walls_and_fluids(self, ):
        walls = []
        fluids = []
        for x in self.scene['walls']:
            points, normals = obj_surface_to_particles(x['path'])
            if 'invert_normals' in x and x['invert_normals']:
                normals = -normals
            points += np.asarray([x['translation']], dtype=np.float32)
            walls.append((points, normals))
        self.box = np.concatenate([x[0] for x in walls], axis=0)
        self.box_normals = np.concatenate([x[1] for x in walls], axis=0)
        
        self.min_y = np.min(self.box[:, 1]) - 0.05 * (np.max(self.box[:, 1]) - np.min(self.box[:, 1]))


        self.box = torch.from_numpy(self.box).to(self.device)   
        self.box_normals = torch.from_numpy(self.box_normals).to(self.device)


        # export static particles
        # write_particles(os.path.join(self.output_dir, 'box'), self.box, self.box_normals)

        # compute lowest point for removing out of bounds particles
        # prepare fluids
        for x in self.scene['fluids']:
            points = obj_volume_to_particles(x['path'])[0]
            points += np.asarray([x['translation']], dtype=np.float32)
            velocities = np.empty_like(points)
            velocities[:, 0] = x['velocity'][0]
            velocities[:, 1] = x['velocity'][1]
            velocities[:, 2] = x['velocity'][2]
            range_ = range(x['start'], x['stop'], x['step'])
            fluids.append((points, velocities, range_))
        return walls, fluids

    
    def step_one(self, pos, vel, step, write_ply=0):

        if write_ply:
            fluid_output_path = os.path.join(self.output_dir, 'fluid_{0:04d}'.format(step))
            if isinstance(pos, np.ndarray):
                write_particles(fluid_output_path, pos, vel)
            else:
                write_particles(fluid_output_path, pos.cpu().numpy(), vel.cpu().numpy(),)
        pos = torch.tensor(pos, device=self.device).float()
        vel = torch.tensor(vel, device=self.device).float()
        inputs = (pos, vel, None, self.box, self.box_normals)
        pos, vel = self.model(inputs)
        pos = pos.cpu().numpy()
        vel = vel.cpu().numpy()
        return [pos, vel]
    

    def steps(self, num_steps=500, write_ply=False):

        pos = np.empty(shape=(0, 3), dtype=np.float32)
        vel = np.empty_like(pos)
        for step in range(num_steps):
            # add from fluids to pos vel arrays
            for points, velocities, range_ in self.fluids:
                if step in range_:  # check if we have to add the fluid at this point in time
                    pos = np.concatenate([pos, points], axis=0).astype(np.float32)
                    vel = np.concatenate([vel, velocities], axis=0).astype(np.float32)

                    pos = torch.tensor(pos, device=self.device)
                    vel = torch.tensor(vel, device=self.device)
            if pos.shape[0]:
                if write_ply:
                    fluid_output_path = os.path.join(self.output_dir, 'fluid_{0:04d}'.format(step))
                    if isinstance(pos, np.ndarray):
                        write_particles(fluid_output_path, pos, vel)
                    else:
                        write_particles(fluid_output_path, pos.cpu().numpy(), vel.cpu().numpy(),)

                inputs = (pos, vel, None, self.box, self.box_normals)
                pos, vel = self.model(inputs)
                
            # remove out of bounds particles
            if step % 10 == 0:
                print(step, 'num particles', pos.shape[0])
            #     mask = pos[:, 1] > self.min_y
            #     if np.count_nonzero(mask) < pos.shape[0]:
            #         pos = pos[mask]
            #         vel = vel[mask]

    def evaluate_whole_sequence(self, dataset_dir):
        from utils import evaluate_whole_sequence_torch
        evaluate_whole_sequence_torch(dataset_dir,
                                    self.model, 
                                    frame_skip = 1,
                                    device="cuda",
                                    scale=1)
if __name__ == "__main__":
    weights_path = "pretrained_model_weights.pt"
    scene_name = "example_scene.json"
    output_dir = "example_out"
    worker = Worker(weights_path, scene_name, output_dir)
    # worker.steps(num_steps=100)
    worker.evaluate_whole_sequence("./datasets")
