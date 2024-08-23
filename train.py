from ultralytics import YOLO
    
model = YOLO('MDRN/ultralytics/cfg/models/MDRN/MDRN.yaml') 

results = model.train(data='ultralytics/cfg/datasets/RSRBD.yaml')


