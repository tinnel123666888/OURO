import cv2
import numpy as np
import easyocr
import re
import time

image_path = "/home/ghc/detectron2-main/ocr_4_elevator/images/image-3-t.jpg"

image = cv2.imread(image_path)

ocr_reader = easyocr.Reader(['en'], gpu=False)  # 适用于中文和英文

ts = time.time()
results = ocr_reader.readtext(image_path, text_threshold=0.7, allowlist="0123456789")
print(f"EasyOCR 识别耗时: {time.time() - ts:.2f} 秒")

ts = time.time()
results = ocr_reader.readtext(image_path, text_threshold=0.7, allowlist="0123456789")

print(f"EasyOCR 识别耗时: {time.time() - ts:.2f} 秒")



ts = time.time()
results = ocr_reader.readtext(image_path, text_threshold=0.7, allowlist="0123456789")

print(f"EasyOCR 识别耗时: {time.time() - ts:.2f} 秒")

ts = time.time()
results = ocr_reader.readtext(image_path, text_threshold=0.7, allowlist="0123456789")

print(f"EasyOCR 识别耗时: {time.time() - ts:.2f} 秒")

digits = []

conf_threshold = 0.8  # 置信度下限
visualize = True  # 是否可视化检测结果
for bbox, text, conf in results:
    # 过滤非数字 & 低置信度
    print(f"检测到文本: {text}, 置信度: {conf}, 边框: {bbox}")
    if conf >= conf_threshold and re.fullmatch(r'\d+', text):
        digits.append((text, conf, bbox))
print(f"识别到 {len(digits)} 个数字 : {digits}")

# # 可视化
# if visualize:
#     img = cv2.imread(image_path)
#     for text, conf, bbox in digits:
#         pts = np.array(bbox).astype(int)
#         cv2.polylines(img, [pts], True, (0, 255, 0), 2)
#         cv2.putText(img, text, tuple(pts[0]),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     cv2.imshow("Digit Detection", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()