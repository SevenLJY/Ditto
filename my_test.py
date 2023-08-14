import os
import numpy as np
import copy
import torch
import open3d as o3d
import json
import numpy as np
from hydra.experimental import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import hydra
from src.utils.joint_estimation import aggregate_dense_prediction_r
from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
from src.utils.metrics import eval_CD_ditto, axis_error, geodesic_distance, euclidean_distance, R_from_axis_angle, combine_pred_mesh
from pytorch_lightning import seed_everything

seed_everything(2)

def sample_point_cloud(pc, num_point):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(
        np.arange(num_point_all),
        size=(num_point,),
        replace=num_point > num_point_all,
    )
    return pc[idxs], idxs

def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)

def vector_to_rotation(vector):
    z = np.array(vector)
    z = z / np.linalg.norm(z)
    x = np.array([1, 0, 0])
    x = x - z*(x.dot(z)/z.dot(z))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.c_[x, y, z]

def save_axis_mesh(k, center, filepath):
    '''support rotate only for now'''
    axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, cylinder_height=1.0, cone_height=0.08)
    arrow = np.array([0., 0., 1.], dtype=np.float32)
    n = np.cross(arrow, k) 
    rad = np.arccos(np.dot(arrow, k))
    R_arrow = R_from_axis_angle(n, rad)
    axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(filepath, axis)

with initialize(config_path='configs/'):
    config = compose(
        config_name='config',
        overrides=[
            'experiment=Ditto_s2m.yaml',
        ], return_hydra_config=True)
config.datamodule.opt.train.data_dir = 'data/'
config.datamodule.opt.val.data_dir = 'data/'
config.datamodule.opt.test.data_dir = 'data/'

# load model
model = hydra.utils.instantiate(config.model)
ckpt = torch.load('pretrained/Ditto_s2m.ckpt')
device = torch.device(0)
model.load_state_dict(ckpt['state_dict'], strict=True)
model = model.eval().to(device)

generator = Generator3D(
    model.model,
    device=device,
    threshold=0.4,
    seg_threshold=0.5,
    input_type='pointcloud',
    refinement_step=0,
    padding=0.1,
    resolution0=32
)

def normalize_pcd(pc_start, pc_end):
    point_start = o3d.utility.Vector3dVector(pc_start)
    point_end = o3d.utility.Vector3dVector(pc_end)
    pcd_start = o3d.geometry.PointCloud(point_start)
    pcd_end = o3d.geometry.PointCloud(point_end)


    b_max_start = pcd_start.get_max_bound()
    b_max_end = pcd_end.get_max_bound()
    b_min_start = pcd_start.get_min_bound()
    b_min_end = pcd_end.get_min_bound()

    b_max = np.maximum(b_max_start, b_max_end)
    b_min = np.minimum(b_min_start, b_min_end)

    scale = (np.max(b_max, 0) - np.min(b_min, 0)).max() * 1.1

    b_center = 0.5 * (b_max + b_min)

    pcd_start.translate(-b_center)
    pcd_end.translate(-b_center)
    
    pcd_start.scale(1./scale, np.zeros((3, 1)))
    pcd_end.scale(1./scale, np.zeros((3, 1)))

    # new downsample
    pcd_start = pcd_start.voxel_down_sample(voxel_size=0.01)
    pcd_end = pcd_end.voxel_down_sample(voxel_size=0.01)

    npc_start = np.asarray(pcd_start.points)
    npc_end = np.asarray(pcd_end.points)

    npc_start, _ = sample_point_cloud(npc_start, 8192)
    npc_end, _ = sample_point_cloud(npc_end, 8192)

    return npc_start, npc_end, scale, b_center

    # R = np.array([
    #     [0., -1., 0.],
    #     [1., 0., 0.],
    #     [0., 0., 1.],
    # ])

    # pcd_start.rotate(R, np.zeros((3, 1)))
    # pcd_end.rotate(R, np.zeros((3, 1)))


    # o3d.io.write_point_cloud('my_laptop_start_rotate.ply', pcd_start)
    # o3d.io.write_point_cloud('my_laptop_end_rotate.ply', pcd_end)


    return np.asarray(pcd_start.points), np.asarray(pcd_end.points), scale, b_center

def test(model_id):
    data_root = '/localhome/jla861/Documents/SFU/Research/Articulated-NeRF/AN3'
    exp_dir = os.path.join('test_results', model_id)
    os.makedirs(exp_dir, exist_ok=True)
    # load npz data
    data = np.load(os.path.join(data_root, 'data/sapien', model_id, 'test_ditto.npz'))

    pc_start = data['pc_start']
    pc_end = data['pc_end']

    pc_start, pc_end, scale, center = normalize_pcd(pc_start, pc_end)

    sample = {
        'pc_start': torch.from_numpy(pc_start).unsqueeze(0).to(device).float(),
        'pc_end': torch.from_numpy(pc_end).unsqueeze(0).to(device).float()
    }

    # inference
    mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(sample)
    with torch.no_grad():
        joint_type_logits, joint_param_revolute, joint_param_prismatic = model.model.decode_joints(mobile_points_all, c)

    # compute articulation model
    mesh_dict[1].visual.face_colors = np.array([84, 220, 83, 255], dtype=np.uint8)
    joint_type_prob = joint_type_logits.sigmoid().mean()

    if joint_type_prob.item()< 0.5:
        # axis voting
        joint_r_axis = (
            normalize(joint_param_revolute[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_r_t = joint_param_revolute[:, :, 3][0].cpu().numpy()
        joint_r_p2l_vec = (
            normalize(joint_param_revolute[:, :, 4:7], -1)[0].cpu().numpy()
        )
        joint_r_p2l_dist = joint_param_revolute[:, :, 7][0].cpu().numpy()
        p_seg = mobile_points_all[0].cpu().numpy()

        pivot_point = p_seg + joint_r_p2l_vec * joint_r_p2l_dist[:, np.newaxis]
        (
            joint_axis_pred,
            pivot_point_pred,
            config_pred,
        ) = aggregate_dense_prediction_r(
            joint_r_axis, pivot_point, joint_r_t, method="mean"
        )
        pred_type = 'rotate'
    # prismatic
    else:
        # axis voting
        joint_p_axis = (
            normalize(joint_param_prismatic[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_axis_pred = joint_p_axis.mean(0)
        joint_p_t = joint_param_prismatic[:, :, 3][0].cpu().numpy()
        config_pred = joint_p_t.mean()
        
        pivot_point_pred = mesh_dict[1].bounds.mean(0)
        pred_type = 'translate'
    
    

    pivot_point_pred = np.cross(joint_axis_pred, np.cross(pivot_point_pred, joint_axis_pred))
    pivot_point_pred = scale * pivot_point_pred + center
    pivot_point_pred = pivot_point_pred + np.dot(joint_axis_pred, -pivot_point_pred) * joint_axis_pred

    static_part = mesh_dict[0].copy().as_open3d
    static_part.scale(scale, np.zeros((3, 1)))
    static_part.translate(center)

    mobile_part = mesh_dict[1].copy().as_open3d
    mobile_part.scale(scale, np.zeros((3, 1)))
    mobile_part.translate(center)


    # o3d.io.write_triangle_mesh(os.path.join(exp_dir, 'static.obj'), static_part)
    o3d.io.write_triangle_mesh(os.path.join(exp_dir, 'static.ply'), static_part)
    # o3d.io.write_triangle_mesh(os.path.join(exp_dir, 'mobile.obj'), mobile_part)
    o3d.io.write_triangle_mesh(os.path.join(exp_dir, 'mobile.ply'), mobile_part)
    

    # save_axis_mesh(joint_axis_pred, pivot_point_pred, os.path.join(exp_dir, 'axis.ply'))
    # save_axis_mesh(-joint_axis_pred, pivot_point_pred, os.path.join(exp_dir, 'axis_oppo.ply'))

    # # canonical mobile part
    # if pred_type == 'rotate':
    #     R_can_gt = R_from_axis_angle(data['joint_axis'], np.deg2rad(data['joint_state']))
    #     center_can_gt = data['joint_pivot']
    #     R_can_pred = R_from_axis_angle(joint_axis_pred, config_pred * 0.5)
    #     center_can_pred = pivot_point_pred
        
    #     can_mobile_w_gt = copy.deepcopy(mobile_part).rotate(R_can_gt, center=center_can_gt)
    #     can_mobile_w_pred = copy.deepcopy(mobile_part).rotate(R_can_pred, center=center_can_pred)
    #     o3d.io.write_triangle_mesh(os.path.join(exp_dir, 'mobile_can_gt.obj'), can_mobile_w_gt)
    #     o3d.io.write_triangle_mesh(os.path.join(exp_dir, 'mobile_can_pred.obj'), can_mobile_w_pred)


    # synmetric chamfer-l1 distance
    gt_static_path = os.path.join(data_root, os.path.dirname(os.path.dirname(data['static_path'].item())), 'start', 'start_static_rotate.ply')
    gt_mobile_path = os.path.join(data_root, os.path.dirname(os.path.dirname(data['mobile_path'].item())), 'start', 'start_dynamic_rotate.ply')
    gt_whole_path = os.path.join(data_root, os.path.dirname(os.path.dirname(data['mobile_path'].item())), 'start', 'start_rotate_resave.ply')


    # gt_static_path = os.path.join(data_root, data['static_path'].item()[:-4] + '.ply')
    # gt_mobile_path = os.path.join(data_root, data['mobile_path'].item()[:-4] + '.ply')
    # gt_whole_path = os.path.join(data_root, os.path.dirname(data['mobile_path'].item()), 'start_rotate_resave.ply')

    pred_static_path = os.path.join(exp_dir, 'static.ply')
    pred_mobile_path = os.path.join(exp_dir, 'mobile.ply')
    pred_whole_path = os.path.join(exp_dir, 'whole.ply')
    combine_pred_mesh([pred_static_path,pred_mobile_path], pred_whole_path)
    cd_s, cd_d, cd_w = eval_CD_ditto(pred_static_path, pred_mobile_path, pred_whole_path, gt_static_path, gt_mobile_path, gt_whole_path)
    
    # joint axis error
    gt_type = data['motion_type']
    gt_axis = data['joint_axis']
    gt_pivot = data['joint_pivot']
    gt_state = data['joint_state'] * 2.

    if gt_type == 'translate':
        save_axis_mesh(joint_axis_pred, gt_pivot, os.path.join(exp_dir, 'axis.ply'))
        save_axis_mesh(-joint_axis_pred, gt_pivot, os.path.join(exp_dir, 'axis_oppo.ply'))
    else:
        save_axis_mesh(joint_axis_pred, pivot_point_pred, os.path.join(exp_dir, 'axis.ply'))
        save_axis_mesh(-joint_axis_pred, pivot_point_pred, os.path.join(exp_dir, 'axis_oppo.ply'))

    ang_err, pos_err = axis_error(joint_axis_pred, gt_axis, pivot_point_pred, gt_pivot)

    # joint state error
    if pred_type == 'rotate':
        state_dist = geodesic_distance(joint_axis_pred, config_pred, gt_axis, np.deg2rad(gt_state))
    else:
        state_dist = euclidean_distance(config_pred, joint_axis_pred, gt_state, gt_axis)
    
    # save the metrics
    with open(os.path.join(exp_dir, 'metrics.txt'), 'w') as f:
        f.write(f'---------- geometry ----------\n')
        f.write(f'CD_static: {cd_s}\n')
        f.write(f'CD_dynamic: {cd_d}\n')
        f.write(f'CD_whole: {cd_w}\n')
        f.write(f'---------- motion ----------\n')
        f.write(f'GT motion type: {gt_type}\n')
        f.write(f'Pred motion type: {pred_type}\n')
        f.write(f'axis angular error: {ang_err}\n')
        f.write(f'axis distance error: {pos_err}\n')
        f.write(f'geodesic/euclidean distance: {state_dist}\n')

if __name__ == '__main__':
    model_ids = [
        # 'storage/45135',
        # 'laptop/10211_easier_tex',
        # 'stapler/103111',
        # 'USB/100109',
        # 'scissor/11100',
        # 'foldingchair/102255',
        # 'fridge/10905_easy',
        # 'blade/103706',
        # 'oven/101917',
        # 'washmachine/103776',
        'fridge/10905_close',
        # 'foldingchair/100520',

    ]

    for model_id in model_ids:
        test(model_id)
        print('finish ', model_id)