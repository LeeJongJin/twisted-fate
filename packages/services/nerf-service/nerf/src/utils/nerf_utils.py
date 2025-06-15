import torch
import numpy as np
import cv2
from typing import Tuple, List

def get_rays(H: int, W: int, K: np.ndarray, c2w: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """카메라 파라미터로부터 광선 생성"""
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def render_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, near: float, far: float,
                N_samples: int, model: torch.nn.Module, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """광선을 따라 샘플링하고 NeRF 모델로 렌더링"""
    # 광선을 따라 샘플링
    z_vals = torch.linspace(near, far, N_samples)
    z_vals = z_vals.expand([rays_o.shape[0], N_samples])
    
    # 샘플링된 3D 포인트 계산
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    
    # NeRF 모델로 RGB와 밀도 예측
    raw = model(pts)
    rgb = torch.sigmoid(raw[...,:3])
    alpha = 1.-torch.exp(-torch.relu(raw[...,3]))
    
    # 볼륨 렌더링
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    
    return rgb_map, weights, z_vals

def extract_frames(video_path: str, output_dir: str) -> List[str]:
    """비디오에서 프레임 추출"""
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_path = f"{output_dir}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
        
    cap.release()
    return frame_paths

def create_camera_poses(num_poses: int) -> np.ndarray:
    """360도 카메라 포즈 생성"""
    thetas = np.linspace(0, 2*np.pi, num_poses)
    poses = []
    
    for theta in thetas:
        c2w = np.array([
            [np.cos(theta), 0, -np.sin(theta), 0],
            [0, 1, 0, 0],
            [np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])
        poses.append(c2w)
        
    return np.array(poses) 