#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
detect_digits.py
使用 EasyOCR + OpenCV 从静态图像中检测纯数字文本
"""

import re
import cv2
import numpy as np
import easyocr
import argparse
from pathlib import Path


def extract_digits(image_path: str,
                   lang=('en', 'ch_sim'),
                   visualize: bool = False,
                   conf_threshold: float = 0.3):
    """
    识别图片中的数字

    Parameters
    ----------
    image_path : str
        图片路径
    lang : tuple
        EasyOCR 语言包；纯数字时用 ('en',) 效率更高
    visualize : bool
        是否弹窗展示检测框
    conf_threshold : float
        置信度下限，低于该值的结果将被忽略

    Returns
    -------
    list[tuple[str, float, list[list[int]]]]
        每条返回 (数字文本, 置信度, 边框四点坐标)
    """
    # 初始化 OCR 模型（CPU 即可；如果装了显卡驱动可设置 gpu=True）
    reader = easyocr.Reader(list(lang), gpu=False)

    # OCR 推理；detail=1 返回 (bbox, text, conf)
    results = reader.readtext(image_path, allowlist="0123456789",detail=1, paragraph=False)

    digits = []
    for bbox, text, conf in results:
        # 过滤非数字 & 低置信度
        if conf >= conf_threshold and re.fullmatch(r'\d+', text):
            digits.append((text, conf, bbox))

    # 可视化
    if visualize:
        img = cv2.imread(str(image_path))
        for text, conf, bbox in digits:
            pts = np.array(bbox).astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.putText(img, text, tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Digit Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return digits


def main():
    parser = argparse.ArgumentParser(
        description="Detect digits in an image via EasyOCR + OpenCV")
    parser.add_argument("-image", help="Path to the target image")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Display a window with detection results")
    parser.add_argument("-l", "--lang", default="en",
                        help="Languages for EasyOCR, e.g. 'en', 'ch_sim', "
                             "'en,ch_sim'; digits only -> keep as 'en'")
    parser.add_argument("-t", "--thres", type=float, default=0.3,
                        help="Confidence threshold (0-1)")
    args = parser.parse_args()

    if not Path(args.image).is_file():
        parser.error(f"Image '{args.image}' not found.")

    lang_tuple = tuple(map(str.strip, args.lang.split(',')))
    detections = extract_digits(args.image,
                                lang=lang_tuple,
                                visualize=args.visualize,
                                conf_threshold=args.thres)

    if detections:
        print("Detected digits:")
        for text, conf, _ in detections:
            print(f"  {text}    (confidence = {conf:.2f})")
    else:
        print("No digits detected.")


if __name__ == "__main__":
    main()
