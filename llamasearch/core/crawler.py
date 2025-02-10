import requests
import os

def extract_text_with_jina(url):

    # Use Jina's URL reader service
    jina_reader_url = f"https://r.jina.ai/{url}"#URL ented by user
    
    try:
        response = requests.get(jina_reader_url, timeout=10)
        response.raise_for_status()
        # Jina returns already cleaned text
        return response.text  
    # Handles errors
    except requests.RequestException as e:
        print(f"Error fetching from Jina API: {e}")
        return None

def find_project_root():
    # Finds the root of the project
    current_dir = os.path.abspath(os.path.dirname(__file__)) 

    while current_dir != os.path.dirname(current_dir): 
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir 

        current_dir = os.path.dirname(current_dir)

    raise RuntimeError("Could not find project root. Please check your project structure.")

def save_to_project_tempdir(text, filename="extracted_text.txt"):
#Saves output in the root dir in temp dir, will create one if no temp dir exits
    """Saves text to a temp directory inside the project root."""
    project_root = find_project_root() 
    temp_dir = os.path.join(project_root, "temp")  # Define temp directory path
    
    os.makedirs(temp_dir, exist_ok=True)  # Create the temp dir if it doesn't exist
    file_path = os.path.join(temp_dir, filename)  # Define full file path
    
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    
    return file_path  # Return the file path for reference

if __name__ == "__main__":
    url = input("Enter the webpage URL: ").strip()
    text = extract_text_with_jina(url)

    if text:
        file_path = save_to_project_tempdir(text)
        print(f"\nExtracted content saved at: {file_path}")
