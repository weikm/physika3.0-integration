import os 
import sys
from glob import glob
import subprocess
import tempfile
import open3d as o3d
import numpy as np
import zstandard as zstd
import msgpack
import msgpack_numpy
import dataflow
from scipy.spatial import cKDTree
from splishsplash_config import VOLUME_SAMPLING_BIN
msgpack_numpy.patch()
PARTICLE_RADIUS = 0.025

class PhysicsSimDataFlow(dataflow.RNGDataFlow):
    """Data flow for msgpacks generated from SplishSplash simulations.
    """

    def __init__(self, files, random_rotation=False, shuffle=False, window=2):
        if not len(files):
            raise Exception("List of files must not be empty")
        if window < 1:
            raise Exception("window must be >=1 but is {}".format(window))
        self.files = files
        self.random_rotation = random_rotation
        self.shuffle = shuffle
        self.window = window

    def __iter__(self):
        decompressor = zstd.ZstdDecompressor()
        files_idxs = np.arange(len(self.files))
        if self.shuffle:
            self.rng.shuffle(files_idxs)

        for file_i in files_idxs:
            # read all data from file
            with open(self.files[file_i], 'rb') as f:
                data = msgpack.unpackb(decompressor.decompress(f.read()),
                                       raw=False)

            data_idxs = np.arange(len(data) - self.window + 1)
            if self.shuffle:
                self.rng.shuffle(data_idxs)

            # get box from first item. The box is valid for the whole file
            box = data[0]['box']
            box_normals = data[0]['box_normals']

            for data_i in data_idxs:

                if self.random_rotation:
                    angle_rad = self.rng.uniform(0, 2 * np.pi)
                    s = np.sin(angle_rad)
                    c = np.cos(angle_rad)
                    rand_R = np.array([c, 0, s, 0, 1, 0, -s, 0, c],
                                      dtype=np.float32).reshape((3, 3))

                if self.random_rotation:
                    sample = {
                        'box': np.matmul(box, rand_R),
                        'box_normals': np.matmul(box_normals, rand_R)
                    }
                else:
                    sample = {'box': box, 'box_normals': box_normals}

                for time_i in range(self.window):

                    item = data[data_i + time_i]

                    for k in ('pos', 'vel'):
                        if self.random_rotation:
                            sample[k + str(time_i)] = np.matmul(item[k], rand_R)
                        else:
                            sample[k + str(time_i)] = item[k]

                    for k in ('m', 'viscosity', 'frame_id', 'scene_id'):
                        sample[k + str(time_i)] = item[k]

                yield sample


def read_data_val(files, **kwargs):
    return read_data(files=files,
                     batch_size=1,
                     repeat=False,
                     shuffle_buffer=None,
                     num_workers=1,
                     **kwargs)


def read_data_train(files, batch_size, random_rotation=True, **kwargs):
    return read_data(files=files,
                     batch_size=batch_size,
                     random_rotation=random_rotation,
                     repeat=True,
                     shuffle_buffer=512,
                     **kwargs)


def read_data(files=None,
              batch_size=1,
              window=2,
              random_rotation=False,
              repeat=False,
              shuffle_buffer=None,
              num_workers=1,
              cache_data=False):
    print(files[0:20], '...' if len(files) > 20 else '')

    # caching makes only sense if the data is finite
    if cache_data:
        if repeat == True:
            raise Exception("repeat must be False if cache_data==True")
        if random_rotation == True:
            raise Exception("random_rotation must be False if cache_data==True")
        if num_workers != 1:
            raise Exception("num_workers must be 1 if cache_data==True")

    df = PhysicsSimDataFlow(
        files=files,
        random_rotation=random_rotation,
        shuffle=True if shuffle_buffer else False,
        window=window,
    )

    if repeat:
        df = dataflow.RepeatedData(df, -1)

    if shuffle_buffer:
        df = dataflow.LocallyShuffleData(df, shuffle_buffer)

    if num_workers > 1:
        df = dataflow.MultiProcessRunnerZMQ(df, num_proc=num_workers)

    df = dataflow.BatchData(df, batch_size=batch_size, use_list=True)

    if cache_data:
        df = dataflow.CacheData(df)

    df.reset_state()
    return df



def write_particles(path_without_ext, pos, vel=None, options=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    arrs = {'pos': pos}
    if not vel is None:
        arrs['vel'] = vel
    np.savez(path_without_ext + '.npz', **arrs)

    if options and options.write_ply:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos))
        o3d.io.write_point_cloud(path_without_ext + '.ply', pcd)

    if options and options.write_bgeo:
        write_bgeo_from_numpy(path_without_ext + '.bgeo', pos, vel)

def _ground_truth_to_prediction_distance(pred, gt):
    tree = cKDTree(pred)
    dist, _ = tree.query(gt)
    return dist


def _compute_stats(x):
    tmp = {
        'mean': np.mean(x),
        'mse': np.mean(x**2),
        'var': np.var(x),
        'min': np.min(x),
        'max': np.max(x),
        'median': np.median(x),
    }
    tmp = {k: float(v) for k, v in tmp.items()}
    tmp['num_particles'] = x.shape[0]
    return tmp

def evaluate_whole_sequence_torch(dataset_dir, model, frame_skip = 1, device="cuda", scale=1):
    import torch
    print('evaluating.. ', end='')
    sys.path.append('.')
    val_files = sorted(glob(os.path.join(dataset_dir, 'valid', '*.zst')))
    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    model.requires_grad_(False)

    errs = []

    skip = frame_skip

    last_scene_id = None
    for data in val_dataset:
        scene_id = data['scene_id0'][0]
        if last_scene_id is None or last_scene_id != scene_id:
            print(scene_id, end=' ', flush=True)
            last_scene_id = scene_id
            box = torch.from_numpy(data['box'][0].copy()).to(device)
            box_normals = torch.from_numpy(data['box_normals'][0].copy()).to(device)
            init_pos = torch.from_numpy(data['pos0'][0].copy()).to(device)
            init_vel = torch.from_numpy(data['vel0'][0].copy()).to(device)

            inputs = (init_pos, init_vel, None, box, box_normals)
        else:
            inputs = (pr_pos, pr_vel, None, box, box_normals)

        pr_pos, pr_vel = model(inputs)

        frame_id = data['frame_id0'][0]
        if frame_id > 0 and frame_id % skip == 0:
            gt_pos = data['pos0'][0]
            gt_to_pred_distances = _ground_truth_to_prediction_distance(pr_pos.cpu().numpy(), gt_pos)
            gt_to_pred_errs = _compute_stats(gt_to_pred_distances)
            errs.append(gt_to_pred_errs.get("mean"))
    print("whole sequence errs is ", np.mean(errs))
    plot_errs(errs)
    print('done')

def plot_errs(errs):
    import matplotlib.pyplot as plt
    plt.figure()  # 创建一个新的图形
    plt.plot(errs)  # 绘制折线图
    plt.xlabel('Step')
    plt.ylabel('gt2pred_mean')
    plt.savefig(f'gt2pred_mean.png')  # 保存图形
    plt.close()  # 关闭当前图形，以便开始下一个
    print("plot the line chart done!")

def create_model():
    # print("In create_model")
    from default_torch import MyParticleNetwork
    # print("import MyParticleNetwork")
    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork()
    # print("get the model")
    return model

def numpy_from_bgeo(path):
    import partio
    p = partio.read(path)
    pos = p.attributeInfo('position')
    vel = p.attributeInfo('velocity')
    ida = p.attributeInfo('trackid')  # old format
    if ida is None:
        ida = p.attributeInfo('id')  # new format after splishsplash update
    n = p.numParticles()
    pos_arr = np.empty((n, pos.count))
    for i in range(n):
        pos_arr[i] = p.get(pos, i)

    vel_arr = None
    if not vel is None:
        vel_arr = np.empty((n, vel.count))
        for i in range(n):
            vel_arr[i] = p.get(vel, i)

    if not ida is None:
        id_arr = np.empty((n,), dtype=np.int64)
        for i in range(n):
            id_arr[i] = p.get(ida, i)[0]

        s = np.argsort(id_arr)
        result = [pos_arr[s]]
        if not vel is None:
            result.append(vel_arr[s])
    else:
        result = [pos_arr, vel_arr]

    return tuple(result)


def obj_volume_to_particles(objpath, scale=1, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, 'out.bgeo')
        scale_str = '{0}'.format(scale)
        radius_str = str(radius)
        status = subprocess.run([
            VOLUME_SAMPLING_BIN, '-i', objpath, '-o', outpath, '-r', radius_str,
            '-s', scale_str
        ])
        return numpy_from_bgeo(outpath)


def obj_surface_to_particles(objpath, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    obj = o3d.io.read_triangle_mesh(objpath)
    particle_area = np.pi * radius**2
    # 1.9 to roughly match the number of points of SPlisHSPlasHs surface sampling
    num_points = int(1.9 * obj.get_surface_area() / particle_area)
    pcd = obj.sample_points_poisson_disk(num_points, use_triangle_normal=True)
    points = np.asarray(pcd.points).astype(np.float32)
    normals = -np.asarray(pcd.normals).astype(np.float32)
    return points, normals
