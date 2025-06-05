import json
import os
import argparse
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Process all root directories under a given dataset root.")
    parser.add_argument('--dataset_root', type=str, required=True, help="Path to the root directory containing all root directories (e.g., /your_dataset_root/)")
    parser.add_argument('--output_file', type=str, required=True, help="Output JSON file path to save merged captions")
    parser.add_argument('--similarity_threshold', type=float, default=0.2, help="Cosine similarity threshold")
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
    Build a tree: each node is a dir, value is {'img': img_path, 'desc': desc, 'children': [...]}
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
    return node_map, roots

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
    if curr_desc:
        if child_merged_block:
            merged_result = merge_to_parent(curr_desc, child_merged_block, threshold)
        else:
            merged_result = curr_desc
    else:
        merged_result = child_merged_block
    return merged_result

def main():
    args = parse_args()
    output = []

    # 1. Automatically traverse all root directories under dataset_root
    for root_dir in os.listdir(args.dataset_root):
        root_path = os.path.join(args.dataset_root, root_dir)
        if os.path.isdir(root_path):
            data_file = os.path.join(root_path, 'data.json')
            if os.path.exists(data_file):
                print(f"Processing root directory: {root_path}")
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 2. Build tree structure for this root directory
                node_map, _ = build_tree(data)
                
                # 3. Recursively merge descriptions for this root directory
                merged_desc = recursive_merge_dir(node_map, root_path, args.similarity_threshold)
                root_img = node_map[root_path]['img']
                
                # 4. Add the result to output
                output.append({
                    "root_image_path": root_img,
                    "final_description": merged_desc
                })

    # 5. Save the merged descriptions to the output file
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nMerged descriptions saved to {args.output_file}")

if __name__ == "__main__":
    main()
