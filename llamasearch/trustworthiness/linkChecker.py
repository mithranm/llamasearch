import sys
import importlib
import tldextract
from pathlib import Path

def extract_domain(url):
    if isinstance(url, str) is False or url.strip() == "":
        raise ValueError("Invalid URL: Input must be a non-empty string.")
    
    ext = tldextract.extract(url)
    # This will return the TLD if found

    if ext.domain == "" or ext.suffix == "":
        raise ValueError(f"Invalid URL format: '{url}'")
    
    return ext.domain

def resolve_md_file_path(md_file):
    """Resolve the absolute path for the markdown file."""
    path = Path(md_file)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "temp" / "links.md"
    return path


def validate_file_exists(path):
    """Check if the file exists and is readable."""
    if not path.exists() or not path.is_file():
        print(f"Error: Markdown file '{path}' does not exist or is not a file.")
        return False
    return True

def check_links_domain(md_file, database):
    # Sets the trusted source links to read from
    trusted_sources = set(database.trustedSources)

    # Initializes the count of trusted sources and the amount read through
    count = 0
    total = 0
    
    with open(md_file, 'r', encoding='utf-8') as f:
        for line in f:
            url = line.strip()

            if not url:
                # skip the blank lines
                continue
            
            # find the domain
            crawler_domain = extract_domain(url)

            # finds if the domain is in the trusted sources
            if crawler_domain in trusted_sources:
                count += 1

            # after finishing add to the total looked through
            total += 1

    return count, total


def check_links_end(md_file):
    count = 0
    total = 0
    
    with open(md_file, 'r', encoding='utf-8') as f:
        for line in f:
            # skip empty lines
            if not line.strip():
                continue
            
            # check to see if the line has a reliable top-level domain (TLD)
            if ".edu" in line or ".gov" in line or ".int" in line:
                count += 1
            # add to the total amount of lines looked through
            total += 1

    return count, total

def create_ratio(count, total):
    # create the ratio to output as a percent and avoid division by 0
    if total > 0:
        ratio = (count/total) * 100
    else:
        ratio = 0

    return ratio
            
def main():
    # defines the command line arguments for testing the functionality of linkChecker.py
    if len(sys.argv) != 3:
        print("Usage: python linkChecker.py <links.md> <trustedSources.py>")
        return

    md_file_input = sys.argv[1]
    db_module_name = sys.argv[2].replace('.py', '')

    # Check the md_file and validate the markdown file
    md_file = resolve_md_file_path(md_file_input)
    if validate_file_exists(md_file) is False:
        return

    try:
        # Dynamically import the database module
        database = importlib.import_module(db_module_name)
    except ModuleNotFoundError:
        print(f"Error: Could not find module '{db_module_name}'. Make sure the .py file is in the same directory.")
        return
    except AttributeError:
        print(f"Error: The module '{db_module_name}' does not contain 'trustedSources'. Ensure that the variable is present.")
        return

    # Call your checking functions
    count_domain, total_domain = check_links_domain(md_file, database)
    count_tld, total_tld = check_links_end(md_file)
    domain_ratio = create_ratio(count_domain, total_domain)
    tld_ratio = create_ratio(count_tld, total_tld)

    # Print results
    print(f"Percentage of links from trusted domains: {domain_ratio:.2f}%")
    print(f"Percentage of links with trusted TLDs (.edu/.gov/.int): {tld_ratio:.2f}%")

if __name__ == "__main__":
    main()