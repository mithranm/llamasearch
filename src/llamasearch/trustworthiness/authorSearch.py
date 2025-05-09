import os
import json

def create_ratio(numerator, denominator):
    """
    Create a ratio to output as a percent and avoid division by 0.
    
    Args:
        numerator (int): The numerator of the ratio.
        denominator (int): The denominator of the ratio.

    Returns:
        float: The ratio as a percentage.
    """
    if denominator > 0:
        ratio = (numerator / denominator) * 100
    else:
        ratio = 0

    return ratio

def find_author_trustworthiness_score():
    """
    This function calculates the trustworthiness score of an author based of if the author is found
    through crawling. 
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the crawl_data/domain_authors.json file
    crawl_data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'crawl_data'))
    json_file_path = os.path.join(crawl_data_dir, 'domain_authors.json')

    # Check if the file exists
    if not os.path.isfile(json_file_path):
        print(f"Error: {json_file_path} not found.")
        return 0.0  # Default to not trustworthy if the file is missing

    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        trusted_sources = json.load(f)
    
    # Count the total number of keys and the number of keys with a valid author
    total_keys = len(trusted_sources)
    valid_authors_count = sum(
        1 for key, value in trusted_sources.items() if value.get("author") is not None
    )

    print(f"Total keys: {total_keys}")
    print(f"Keys with valid authors: {valid_authors_count}")
    # Calculate the trustworthiness score
    score = create_ratio(valid_authors_count, total_keys)
    # Print the score
    print(f"Author trustworthiness score: {score:.2f}%")

    return score
    
def main():
    find_author_trustworthiness_score()

if __name__ == "__main__":
    main()