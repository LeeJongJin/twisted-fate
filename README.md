# Twisted Fate

NeRF(Neural Radiance Fields)를 이용한 3D 모델링 웹 애플리케이션

## 프로젝트 구조

```
packages/
  ├── services/
  │   └── nerf-service/
  │       ├── api/           # NestJS API 서버
  │       ├── web/           # React 웹 애플리케이션
  │       └── nerf/          # NeRF 모델 구현
  └── shared/                # 공통 코드
```

## 기술 스택

### Frontend
- React
- TypeScript
- Three.js
- Vite

### Backend
- Node.js
- NestJS
- TypeScript

### NeRF
- PyTorch
- CUDA
- OpenCV

## 설치 방법

### 필수 요구사항
- Node.js 16+
- Python 3.8+
- CUDA Toolkit 11.8+
- NVIDIA GPU

### 설치 단계

1. 저장소 클론
```bash
git clone https://github.com/your-username/twisted-fate.git
cd twisted-fate
```

2. 의존성 설치
```bash
# 루트 디렉토리
npm install

# API 서버
cd packages/services/nerf-service/api
npm install

# 웹 애플리케이션
cd ../web
npm install

# NeRF 모델
cd ../nerf
pip install -r requirements.txt
```

3. 환경 변수 설정
```bash
# .env 파일 생성
cp .env.example .env
```

## 실행 방법

1. 개발 서버 실행
```bash
npm run dev
```

2. 프로덕션 빌드
```bash
npm run build
npm start
```

## 라이선스

MIT 