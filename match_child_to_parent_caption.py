import json
import os
import argparse
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def parse_args():
    """Parse command line arguments for input file, output file, and threshold."""
    parser = argparse.ArgumentParser(description="Match child captions to parent captions based on similarity")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file containing image descriptions")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSON file with updated captions")
    parser.add_argument('--similarity_threshold', type=float, default=0.2, help="Cosine similarity threshold for child-to-parent caption matching (default is 0.2)")
    return parser.parse_args()

def find_deepest_level(image_hierarchy):
    """Find the deepest level of paths in the hierarchy."""
    return max(len(path.split("/")) for path in image_hierarchy)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load the input JSON file
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Read the JSON data

    # Initialize data structures
    image_hierarchy = defaultdict(list)
    image_descriptions = {}

    # Extract hierarchy and descriptions from the data
    for image_group in data:
        for image_list in image_group:
            for img in image_list:
                image_path = img["image_path"]
                parent_path = os.path.dirname(image_path)  # Get the parent directory
                image_hierarchy[parent_path].append(image_path)
                image_descriptions[image_path] = img["answer"]

    # Find the deepest level in the hierarchy
    max_depth = find_deepest_level(image_hierarchy)

    # Traverse and match child captions to parent captions starting from the deepest level
    for depth in range(max_depth, 1, -1):  # Iterate from the deepest level to level 1
        for parent, children in image_hierarchy.items():
            if len(parent.split("/")) == depth:  # Process only the current level
                # Get descriptions for all child images
                child_texts = [image_descriptions[child] for child in children if child in image_descriptions]
                parent_texts = [image_descriptions[parent]] if parent in image_descriptions else []

                if not child_texts:
                    continue  # Skip if no child descriptions are available
                
                # Use TF-IDF and cosine similarity to match child captions with parent caption
                if parent_texts:
                    vectorizer = TfidfVectorizer().fit(child_texts + parent_texts)
                    tfidf_matrix = vectorizer.transform(child_texts + parent_texts)
                    similarity_matrix = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])  # Compute similarity

                    # Select the child description with the highest similarity
                    best_match_index = similarity_matrix.argmax()
                    best_match_text = child_texts[best_match_index]

                    # If similarity is below the threshold, concatenate all child descriptions
                    if similarity_matrix[best_match_index][0] < args.similarity_threshold:
                        best_match_text = " ".join(child_texts)

                    # Append the best matching child caption to the parent caption
                    if parent in image_descriptions:
                        image_descriptions[parent] += " " + best_match_text
                    else:
                        image_descriptions[parent] = best_match_text
                else:
                    # If parent has no description, merge all child descriptions
                    image_descriptions[parent] = " ".join(child_texts)

    # Update the JSON data with new captions
    for image_group in data:
        for image_list in image_group:
            for img in image_list:
                image_path = img["image_path"]
                if image_path in image_descriptions:
                    img["answer"] = image_descriptions[image_path]  # Update the description

    # Save the updated JSON data to the output file
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Processing complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
