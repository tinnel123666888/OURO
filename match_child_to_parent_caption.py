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

def sentence_similarity(sentences1, sentence2):
    """Calculate similarity between a sentence and a list of sentences using TF-IDF and cosine similarity."""
    vectorizer = TfidfVectorizer().fit(sentences1 + [sentence2])
    tfidf_matrix = vectorizer.transform(sentences1 + [sentence2])
    similarity_matrix = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])  # Compute similarity with each sentence
    return similarity_matrix

def merge_child_descriptions_to_parent(parent, children, image_descriptions):
    """Merge child descriptions to parent description based on similarity."""
    parent_text = image_descriptions[parent]
    parent_sentences = parent_text.split(". ")  # Split parent description into sentences

    # Collect descriptions for all child images
    child_texts = [image_descriptions[child] for child in children if child in image_descriptions]
    
    if not child_texts:
        return parent_text  # Return original parent if no children

    # For each child description, match it to parent sentences
    for child_text in child_texts:
        similarity_matrix = sentence_similarity(parent_sentences, child_text)  # Compute similarity

        # If similarity is above the threshold, replace the parent sentence
        for i, parent_sentence in enumerate(parent_sentences):
            if similarity_matrix[0][i] > 0.2:  # Compare similarity with each parent sentence
                parent_sentences[i] = child_text  # Replace with child description
                break
        else:
            # If no match is found, append the child description to the parent
            parent_sentences.append(child_text)

    # Join the updated sentences back into a full description
    return ". ".join(parent_sentences)

def process_node_and_descendants(image_hierarchy, image_descriptions, parent_path):
    """Recursively merge child descriptions into parent description and process all layers."""
    # Process the current level (parent node)
    if parent_path not in image_descriptions:
        return  # Skip if no description exists for the parent

    children = image_hierarchy[parent_path]
    if not children:
        return  # Skip if no children exist

    # Merge all child descriptions into the parent description
    merged_description = merge_child_descriptions_to_parent(parent_path, children, image_descriptions)
    image_descriptions[parent_path] = merged_description

    # Recursively process all children (if they have their own children)
    for child in children:
        process_node_and_descendants(image_hierarchy, image_descriptions, child)

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

    # Process all nodes starting from the topmost parent
    root_nodes = [node for node in image_hierarchy if len(node.split("/")) == 1]
    for root in root_nodes:
        process_node_and_descendants(image_hierarchy, image_descriptions, root)

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
