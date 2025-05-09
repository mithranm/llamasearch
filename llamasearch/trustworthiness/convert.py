import os

def convert_to_md(file_path, output_path):
    """
    Convert a text file of links to a markdown file.

    Args:
        file_path (str): The path to the input text file.
        output_path (str): The path to the output markdown file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()

    md_lines = []
    for line in content:
        link = line.strip()
        if link:
            md_lines.append(f"{link}\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)

def main():
    # Get the directory where this script is located (trustworthiness/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up two levels to llamasearch/
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # Input path: llamasearch/crawl_data/links.txt
    input_file = os.path.join(root_dir, 'crawl_data', 'links.txt')

    # Output path: trustworthiness/links.md
    output_file = os.path.join(current_dir, 'links.md')

    if not os.path.isfile(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    convert_to_md(input_file, output_file)
    print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    main()