import os
import torch
import numpy as np
from models.nerf import NeRFModel
from utils.nerf_utils import get_rays, render_rays, create_camera_poses
import cv2

def render_360_view(model_path: str, output_dir: str, num_views: int = 360):
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeRFModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 카메라 파라미터 설정
    H, W = 512, 512
    focal = 1000.0
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    
    # 360도 카메라 포즈 생성
    poses = create_camera_poses(num_views)
    
    # 각 포즈에서 렌더링
    with torch.no_grad():
        for i, pose in enumerate(poses):
            # 광선 생성
            rays_o, rays_d = get_rays(H, W, K, pose)
            rays_o = torch.from_numpy(rays_o).to(device)
            rays_d = torch.from_numpy(rays_d).to(device)
            
            # 렌더링
            rgb_pred, _, _ = render_rays(rays_o, rays_d, near=0.0, far=1.0,
                                       N_samples=64, model=model)
            
            # 이미지로 변환
            image = (rgb_pred.cpu().numpy() * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 저장
            output_path = os.path.join(output_dir, f'view_{i:04d}.jpg')
            cv2.imwrite(output_path, image)
            
            print(f'Rendered view {i+1}/{num_views}')

def create_video(image_dir: str, output_path: str, fps: int = 30):
    """렌더링된 이미지들을 비디오로 변환"""
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    if not images:
        raise ValueError("No images found in the directory")
    
    # 첫 번째 이미지로부터 비디오 설정
    first_image = cv2.imread(os.path.join(image_dir, images[0]))
    height, width = first_image.shape[:2]
    
    # 비디오 작성기 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 이미지들을 비디오에 추가
    for image_name in images:
        image_path = os.path.join(image_dir, image_name)
        frame = cv2.imread(image_path)
        out.write(frame)
    
    out.release()
    print(f'Video saved to {output_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save rendered views')
    parser.add_argument('--num_views', type=int, default=360,
                      help='Number of views to render')
    parser.add_argument('--fps', type=int, default=30,
                      help='FPS for output video')
    args = parser.parse_args()
    
    # 360도 뷰 렌더링
    render_360_view(args.model_path, args.output_dir, args.num_views)
    
    # 비디오 생성
    video_path = os.path.join(args.output_dir, '360_view.mp4')
    create_video(args.output_dir, video_path, args.fps) 