import os
from ultralytics import RTDETR

# 파일 존재 확인
print("rtdetr-x.pt 존재:", os.path.exists("rtdetr-x.pt"))
print("data.yaml 존재:", os.path.exists("data.yaml"))

try:
    print("모델 로딩 중...")
    model = RTDETR("rtdetr-x.pt")
    print("모델 로딩 완료")
    
    print("훈련 시작...")
    results = model.train(
        data="data.yaml", 
        epochs=1000, 
        imgsz=640, 
        batch=20,
        verbose=True  # 더 많은 출력
    )
    print("훈련 완료")
    
except Exception as e:
    print(f"에러 발생: {e}")
    import traceback
    traceback.print_exc()