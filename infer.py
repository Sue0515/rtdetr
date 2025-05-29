from ultralytics import RTDETR

model = RTDETR("/root/rtdetr/runs/detect/train3/weights/best.pt")

img = "/root/rtdetr/data/datasets/coco128/images/valid/valid_00035.jpg"  # 또는 절대경로: "/root/rtdetr/input.jpg"

# 추론 실행
results = model(img)

# 결과 저장 및 출력
for i, r in enumerate(results):
    # 바운딩 박스가 그려진 이미지 저장
    r.save(f"result_{i+1}.jpg")
    
    # 검출된 객체 정보 출력
    print(f"\n=== 결과 {i+1} ===")
    print(f"검출된 객체 수: {len(r.boxes)}")
    
    # 각 검출된 객체의 정보
    for j, box in enumerate(r.boxes):
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = model.names[class_id]
        
        print(f"객체 {j+1}: {class_name} (신뢰도: {confidence:.3f})")

print(f"\n결과 이미지가 저장되었습니다!")