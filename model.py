from ultralytics import YOLO
import torch
import os
import multiprocessing

multiprocessing.freeze_support() # Recommended for Windows

if __name__ == '__main__':
    print(torch.cuda.is_available())

    dataset_path = r'C:\Users\lapt1\Downloads\Tunanetra\dataset\data.yaml'
    model = YOLO('yolo11n.pt') 

    with open(dataset_path, 'r') as file:
        print(file.read())

    torch.cuda.empty_cache()
    model.train(
        data=dataset_path,
        epochs=100,
        imgsz=640,
        batch=16,  
        device='cuda',
        patience=10,
        lr0=0.01,
        augment=True,
    )
    model.export(format='onnx', dynamic=True, simplify=True)
    model.export(format='engine', dynamic=True, simplify=True)