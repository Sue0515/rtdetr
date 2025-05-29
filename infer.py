from ultralytics import RTDETR

model = RTDETR("best.pt")
img   = "input.jpg"

results=model(img)

for i,r in enumerate(results):
    r.save(f"box_{i+1}.jpg")
