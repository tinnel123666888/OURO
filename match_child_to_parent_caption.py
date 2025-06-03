import json
import os
import argparse
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Recursive bottom-up caption merging for virtual tree by path.")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--similarity_threshold', type=float, default=0.2)
    return parser.parse_args()

def cosine_sentence_similarity(parent_sentences, child_text):
    if not parent_sentences or not child_text.strip():
        return []
    vectorizer = TfidfVectorizer().fit(parent_sentences + [child_text])
    tfidf_matrix = vectorizer.transform(parent_sentences + [child_text])
    similarity = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])
    return similarity[:, 0]

def merge_to_parent(parent_desc, child_merged_desc, threshold):
    parent_sentences = [s.strip() for s in parent_desc.split('. ') if s.strip()]
    if not parent_sentences:
        return child_merged_desc
    if not child_merged_desc or not child_merged_desc.strip():
        return '. '.join(parent_sentences)
    sims = cosine_sentence_similarity(parent_sentences, child_merged_desc)
    if sims is not None and len(sims) > 0 and sims.max() > threshold:
        best_idx = sims.argmax()
        parent_sentences[best_idx] = child_merged_desc
    else:
        parent_sentences.append(child_merged_desc)
    return '. '.join(parent_sentences)

def build_tree(json_list):
    """
    Build a tree: each node is a dir, value is {'img':img_path, 'desc':desc, 'children':[...]}
    Return: node_map, root_dir
    """
    node_map = dict()  # dir_path -> node
    child_dirs = set()
    img_dir_to_img = dict()
    img_dir_to_desc = dict()
    all_dirs = set()
    for image_group in json_list:
        for image_list in image_group:
            for img in image_list:
                img_path = img['image_path']
                desc = img['answer']
                dir_path = os.path.dirname(img_path)
                img_dir_to_img[dir_path] = img_path
                img_dir_to_desc[dir_path] = desc
                all_dirs.add(dir_path)

    # Initialize all nodes
    for d in all_dirs:
        node_map[d] = {'img': img_dir_to_img.get(d, None),
                       'desc': img_dir_to_desc.get(d, ''),
                       'children': []}

    # Build tree: find parent-child relationships
    for d in all_dirs:
        parent = os.path.dirname(d)
        if parent in all_dirs and parent != d:
            node_map[parent]['children'].append(d)
            child_dirs.add(d)

    # Find the root (not a child of anyone)
    roots = [d for d in all_dirs if d not in child_dirs]
    # Choose the shortest as root (root directory)
    root_dir = sorted(roots, key=lambda x: len(x.split('/')))[0]
    return node_map, root_dir

def recursive_merge_dir(node_map, curr_dir, threshold, depth=0):
    node = node_map[curr_dir]
    # Recursively merge all children
    child_merged_descs = []
    for child_dir in sorted(node['children']):
        desc = recursive_merge_dir(node_map, child_dir, threshold, depth+1)
        if desc and desc.strip():
            child_merged_descs.append(desc.strip())
    child_merged_block = '. '.join(child_merged_descs) if child_merged_descs else ''
    curr_desc = node['desc']
    # Print the path and current merge for debugging
    print("  " * depth + f"DIR: {curr_dir}")
    if curr_desc:
        print("  " * depth + f"  own: {curr_desc}")
    if child_merged_block:
        print("  " * depth + f"  merged_child: {child_merged_block}")
    # Merge logic: child_merged_block merged到curr_desc
    if curr_desc:
        if child_merged_block:
            merged_result = merge_to_parent(curr_desc, child_merged_block, threshold)
        else:
            merged_result = curr_desc
    else:
        merged_result = child_merged_block
    print("  " * depth + f"  => merged: {merged_result}\n")
    return merged_result

def main():
    args = parse_args()
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    node_map, root_dir = build_tree(data)
    merged_desc = recursive_merge_dir(node_map, root_dir, args.similarity_threshold)
    root_img = node_map[root_dir]['img']
    output = [{
        "root_image_path": root_img,
        "final_description": merged_desc
    }]
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Root image:", root_img)
    print("\nFinal merged description:\n", merged_desc)
    print(f"\nResult saved to {args.output_file}")

if __name__ == "__main__":
    main()
