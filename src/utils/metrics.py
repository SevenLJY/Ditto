from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
import torch
import open3d as o3d
import numpy as np


def combine_pred_mesh(paths, exp_path):
    recon_mesh = o3d.geometry.TriangleMesh()
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(path)
        recon_mesh += mesh
    o3d.io.write_triangle_mesh(exp_path, recon_mesh)


def compute_chamfer(recon_pts,gt_pts):
	with torch.no_grad():
		recon_pts = recon_pts.cuda()
		gt_pts = gt_pts.cuda()
		dist,_ = chamfer_distance(recon_pts,gt_pts,batch_reduction=None)
		dist = dist.item()
	return dist

def compute_recon_error(recon_path, gt_path, n_samples=10000, vis=False):
    verts, faces = load_ply(recon_path)
    recon_mesh = Meshes(verts=[verts], faces=[faces])
    verts, faces = load_ply(gt_path)
    gt_mesh = Meshes(verts=[verts], faces=[faces])

    gt_pts = sample_points_from_meshes(gt_mesh, num_samples=n_samples)
    recon_pts = sample_points_from_meshes(recon_mesh, num_samples=n_samples)


    if vis:
        pts = gt_pts.clone().detach().squeeze().numpy()
        gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.io.write_point_cloud("gt_points.ply", gt_pcd)
        pts = recon_pts.clone().detach().squeeze().numpy()
        recon_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.io.write_point_cloud("recon_points.ply", recon_pcd)

    return compute_chamfer(recon_pts, gt_pts)
    
def eval_CD_ditto(pred_s_ply, pred_d_ply, pred_w_ply, gt_s_ply, gt_d_ply, gt_w_ply):
    # compute distance
    chamfer_dist_s = compute_recon_error(pred_s_ply, gt_s_ply, n_samples=10000, vis=False)
    chamfer_dist_d = compute_recon_error(pred_d_ply, gt_d_ply, n_samples=10000, vis=False)
    chamfer_dist_w = compute_recon_error(pred_w_ply, gt_w_ply, n_samples=10000, vis=False)

    return chamfer_dist_s, chamfer_dist_d, chamfer_dist_w

def axis_error(pred_axis, gt_axis, pred_pivot, gt_pivot):

    dot_product = np.dot(pred_axis, gt_axis)
    
    pred_norm = np.linalg.norm(pred_axis)
    gt_norm = np.linalg.norm(gt_axis)
    cos_theta = dot_product / (pred_norm * gt_norm)
    angle = np.rad2deg(np.arccos(np.abs(cos_theta)))

    w = gt_pivot - pred_pivot
    d1_cross_d2 = np.cross(pred_axis, gt_axis)
    dist = np.abs(np.sum(w * d1_cross_d2)) / np.linalg.norm(d1_cross_d2)
    return angle, dist

def R_from_axis_angle(k, theta):
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R

def geodesic_distance(pred_axis, pred_rad, gt_axis, gt_rad):
    pred_R = R_from_axis_angle(pred_axis, pred_rad)
    gt_R = R_from_axis_angle(gt_axis, gt_rad)
    R = np.matmul(pred_R, gt_R.T)
    cos_angle = np.clip((np.trace(R) - 1.0) * 0.5, a_min=-1., a_max=1.)
    angle = np.rad2deg(np.arccos(cos_angle)) 
    return angle

def euclidean_distance(pred_dist, pred_axis, gt_dist, gt_axis):
    dist = np.sqrt(np.sum((pred_dist * pred_axis - gt_dist * gt_axis) ** 2))
    return dist