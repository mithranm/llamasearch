import requests
import os

file_deleted = False 

JINA_API_URL = "https://r.jina.ai/"

def find_project_root():
    """Finds the root of the project by looking for `pyproject.toml`."""
    current_dir = os.path.abspath(os.path.dirname(__file__))

    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir

        current_dir = os.path.dirname(current_dir)

    raise RuntimeError("Could not find project root. Please check your project structure.")

def save_to_project_tempdir(text, filename="extracted_texts.md"):
    """Saves extracted content to a `temp` directory inside the project root."""
    global file_deleted
    
    project_root = find_project_root()
    temp_dir = os.path.join(project_root, "temp")

    os.makedirs(temp_dir, exist_ok=True)  # Create temp dir if it doesn't exist
    file_path = os.path.join(temp_dir, filename)  # Define full file path
    
    if not file_deleted and os.path.exists(file_path):
        os.remove(file_path)
        file_deleted = True 

    # Write new content to the file
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(text + "\n\n" + "="*50 + "\n\n")  # Add separator for readability

    return file_path  # Return file path for reference

def read_links_from_temp():
    """Reads URLs from the saved links file."""
    project_root = find_project_root()
    temp_file = os.path.join(project_root, "temp", "links.md")

    if not os.path.exists(temp_file):
        print("No links found. Run crawler.py first.")
        return []

    with open(temp_file, "r", encoding="utf-8") as f:
        links = [line.strip() for line in f.readlines()]

    return links

def extract_text_with_jina(url):
    """Fetches cleaned text using Jina AI."""
    try:
        response = requests.get(JINA_API_URL + url, timeout=10)
        response.raise_for_status()
        return response.text.strip()
    except requests.RequestException as e:
        print(f"Error fetching from Jina AI: {e}")
        return None

if __name__ == "__main__":
    links = read_links_from_temp()

    if not links:
        print("No links to process.")
        exit()

    for link in links:
        print(f"Extracting content from: {link}")
        text = extract_text_with_jina(link)

        if text:
            file_path = save_to_project_tempdir(f"## {link}\n\n{text}")
            print(f"Extracted content saved at: {file_path}")
