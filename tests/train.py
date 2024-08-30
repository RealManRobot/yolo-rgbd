import torch
from ultralytics import YOLO


def yolo_train_main(data_path="data", epochs=100, batch=1):
    model = YOLO(r"../ultralytics/cfg/models/my-seg.yaml").load(
        "../pre_weights.pt")  # build a new model from scratch
    try:
        torch.set_grad_enabled(True)
        model.train(data=data_path,
                    epochs=epochs,
                    batch=batch,
                    optimizer='SGD',
                    device=0 if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"进程执行错误{str(e)}")


if __name__ == '__main__':
    yolo_train_main(data_path="../datasets/Fang_Kuai/Fang_Kuai.yaml")
