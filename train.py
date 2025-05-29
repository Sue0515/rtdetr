from ultralytics import RTDETR
# pip install ultralytics -U
# wget https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt

model = RTDETR("rtdetr-x.pt")
results = model.train(data="data.yaml", epochs=10000, imgsz=640,batch=20)