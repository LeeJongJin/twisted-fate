import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.nerf import NeRFModel
from utils.nerf_utils import get_rays, render_rays, extract_frames, create_camera_poses
import numpy as np
from tqdm import tqdm

def train(video_path: str, output_dir: str, num_epochs: int = 1000):
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # 비디오에서 프레임 추출
    frame_paths = extract_frames(video_path, frames_dir)
    
    # 카메라 파라미터 설정
    H, W = 512, 512  # 이미지 해상도
    focal = 1000.0   # 초점 거리
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    
    # 카메라 포즈 생성
    num_poses = len(frame_paths)
    poses = create_camera_poses(num_poses)
    
    # NeRF 모델 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeRFModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # TensorBoard 설정
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # 학습 루프
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        
        for i, (pose, frame_path) in enumerate(zip(poses, frame_paths)):
            # 광선 생성
            rays_o, rays_d = get_rays(H, W, K, pose)
            rays_o = torch.from_numpy(rays_o).to(device)
            rays_d = torch.from_numpy(rays_d).to(device)
            
            # 이미지 로드
            target = torch.from_numpy(np.load(frame_path)).to(device)
            
            # 렌더링
            rgb_pred, _, _ = render_rays(rays_o, rays_d, near=0.0, far=1.0,
                                       N_samples=64, model=model)
            
            # 손실 계산
            loss = torch.mean((rgb_pred - target) ** 2)
            total_loss += loss.item()
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 에폭당 평균 손실 기록
        avg_loss = total_loss / len(frame_paths)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # 주기적으로 모델 저장
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
    
    writer.close()
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save outputs')
    parser.add_argument('--num_epochs', type=int, default=1000,
                      help='Number of training epochs')
    args = parser.parse_args()
    
    train(args.video_path, args.output_dir, args.num_epochs) 