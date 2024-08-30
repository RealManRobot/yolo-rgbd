import sys

sys.path.append("..\src")
# sys.path.append(r"..\build\lib.win-amd64-cpython-38\src")

import cv2
from solver import Solver
from interface import YoloRGBD

# 实例化对象
solver = Solver()
rgbd = YoloRGBD()
yolo_weights = r'coco.pt'
solver_weights = r'CDNet.pth'

model, solver_weights = rgbd.gen_model(yolo_weights=yolo_weights, solver=solver,
                                       solver_weights_path=solver_weights)

# 输入源
color_img = cv2.imread("data.jpg")

deep_data_depth_esi = solver.test(color_img)
deep_data3 = cv2.cvtColor(deep_data_depth_esi, cv2.COLOR_GRAY2BGR)
results = rgbd.detect(model, color_img, deep_data3, 0.5)

annotated_frame, names, xyxys, masks, confs = rgbd.backward_handle_output(results)

print(names)
print(xyxys)
print(confs)
cv2.imshow("annotated_frame", annotated_frame)

cv2.waitKey(0)
