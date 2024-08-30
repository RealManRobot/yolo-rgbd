from abc import ABC, abstractmethod


class DetectBase(ABC):

    @staticmethod
    @abstractmethod
    def forward_handle_input(color_frame, depth_frame):
        pass

    @staticmethod
    @abstractmethod
    def gen_model(yolo_weights, solver, solver_weights_path):
        pass

    @staticmethod
    @abstractmethod
    def backward_handle_output(outputs):
        pass

    @staticmethod
    @abstractmethod
    def detect(model, color_img, deep_data3, conf):
        pass

    @staticmethod
    @abstractmethod
    def delete_model(model):
        pass
