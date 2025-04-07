import sys
import importlib
import tldextract
from pathlib import Path

def extract_domain(url):
    ext = tldextract.extract(url)
    # This will return the TLD
    return ext.domain

def check_links_domain(md_file, database):

    # Dynamically resolve md_file if not an absolute path
    if not Path(md_file).is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        md_file = project_root / "temp" / "links.md"

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

    # create the ratio to output as a percent and avoid division by 0
    if total > 0:
        ratio = (count/total) * 100
    else:
        ratio = 0

    return ratio


def check_links_end(md_file):

    # Initializes the count of trusted sources and the amount read through
    count = 0
    total = 0

    # Dynamically resolve md_file if not an absolute path
    if not Path(md_file).is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        md_file = project_root / "temp" / "links.md"
    
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

    # create the ratio to output as a percent and avoid division by 0
    if total > 0:
        ratio = (count/total) * 100
    else:
        ratio = 0

    return ratio
            
def main():
    # defines the command line arguments for testing the functionality of just linkChecker.py
    if len(sys.argv) != 3:
        print("Usage: python linkChecker.py <links.md> <trusted_sources.py>")
        return

    md_file = sys.argv[1]
    db_module_name = sys.argv[2].replace('.py', '')

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
    domain_ratio = check_links_domain(md_file, database)
    tld_ratio = check_links_end(md_file)

    # Print results
    print(f"Percentage of links from trusted domains: {domain_ratio:.2f}%")
    print(f"Percentage of links with trusted TLDs (.edu/.gov/.int): {tld_ratio:.2f}%")

if __name__ == "__main__":
    main()