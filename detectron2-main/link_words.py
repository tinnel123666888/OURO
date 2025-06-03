import json
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取 JSON 文件
input_file = "/path/to/your_json.json"  # 修改为实际 JSON 文件路径
output_file = "/path/to/matched_json.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # 读取 JSON 文件

# 构建层级结构 { 父路径: [子路径列表] }
image_hierarchy = defaultdict(list)
image_descriptions = {}

# 遍历数据，提取层级关系
for image_group in data:
    for image_list in image_group:
        for img in image_list:
            image_path = img["image_path"]
            parent_path = os.path.dirname(image_path)  # 获取父目录
            image_hierarchy[parent_path].append(image_path)
            image_descriptions[image_path] = img["answer"]

# ** 找到最深层的目录 **
def find_deepest_level(image_hierarchy):
    """ 找到最深的路径层级 """
    return max(len(path.split("/")) for path in image_hierarchy)

max_depth = find_deepest_level(image_hierarchy)

# ** 从最深层开始遍历匹配父图 **
for depth in range(max_depth, 1, -1):  # 逐层向上匹配
    for parent, children in image_hierarchy.items():
        if len(parent.split("/")) == depth:  # 只处理当前层级
            # 获取所有子图的描述
            child_texts = [image_descriptions[child] for child in children if child in image_descriptions]
            parent_texts = [image_descriptions[parent]] if parent in image_descriptions else []

            if not child_texts:
                continue  # 跳过没有子描述的项
            
            # ** 使用 TF-IDF + 余弦相似度 匹配父图描述 **
            if parent_texts:
                vectorizer = TfidfVectorizer().fit(child_texts + parent_texts)
                tfidf_matrix = vectorizer.transform(child_texts + parent_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])  # 计算子图与父图的相似度

                # 选择 **相似度最高** 的子图描述
                best_match_index = similarity_matrix.argmax()
                best_match_text = child_texts[best_match_index]

                # ** 如果相似度低于阈值（如 0.2），则直接合并所有子描述**
                if similarity_matrix[best_match_index][0] < 0.2:
                    best_match_text = " ".join(child_texts)

                # ** 追加最匹配的子图描述到父图 **
                if parent in image_descriptions:
                    image_descriptions[parent] += " " + best_match_text
                else:
                    image_descriptions[parent] = best_match_text
            else:
                # **父图没有描述，直接合并所有子描述**
                image_descriptions[parent] = " ".join(child_texts)

# ** 更新 JSON 数据 **
for image_group in data:
    for image_list in image_group:
        for img in image_list:
            image_path = img["image_path"]
            if image_path in image_descriptions:
                img["answer"] = image_descriptions[image_path]  # 更新描述

# ** 保存新 JSON 文件 **
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"处理完成，结果已保存到 {output_file}")
