# Original tested crawler.py url: https://www.gnu.org/home.html?distro=trisquel#gnu-linux
import os
import sys

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import crawler.py
from src.llamasearch.core import crawler
from src.llamasearch.trustworthiness.convert import main as convert_main
from src.llamasearch.trustworthiness.linkChecker import main as link_checker_main
from src.llamasearch.trustworthiness.score import calculate_trustworthiness_score
from src.llamasearch.trustworthiness.authorSearch import find_author_trustworthiness_score

def main():
    # Run the crawler
    links = crawler.smart_crawl(
        start_url="https://www.sciencedirect.com/science/article/pii/S2090123221001491",
        target_links=5,
        max_depth=2
    )

     # Check if the crawler collected links
    if not links:
        print("Crawl failed: No links were collected.")
        return

    # Path to the crawl_data directory
    crawl_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', 'crawl_data'))

    # Input file created by crawler.py
    links_file = os.path.join(crawl_data_dir, 'links.txt')

    # Ensure the input file exists
    if not os.path.isfile(links_file):
        print(f"Error: {links_file} not found.")
        return

    # Run convert.py's main() function
    print("Running convert.py...")
    convert_main()
    print("convert.py completed successfully.")

    # Path to the Markdown file created by convert.py
    md_file = os.path.join(os.path.dirname(__file__), 'links.md')

    # Path to the trustedSources module
    trusted_sources_module = "trustedSources"

    # Simulate command-line arguments for linkChecker.py
    sys.argv = ["linkChecker.py", md_file, trusted_sources_module]

    # Run linkChecker.py's main() function
    print("Running linkChecker.py...")
    result = link_checker_main()

    if result is None or not isinstance(result, (tuple, list)) or len(result) != 2:
        print("Error: linkChecker.py did not return valid domain and TLD ratios.")
        return
    
    domain_ratio, tld_ratio = result
    print(f"LinkChecker.py completed successfully.")

    # Get authors trustworthiness score
    print("Calculating author trustworthiness score...")
    author_trustworthiness_score  = find_author_trustworthiness_score()

    # Calculate the linkChecker trustworthiness score
    print("Calculating link trustworthiness score...")
    link_trustworthiness_score = calculate_trustworthiness_score(domain_ratio, tld_ratio, 0.7, 0.3)

    # Calculate the final trustworthiness score
    print("Calculating final trustworthiness score...")
    avg = calculate_trustworthiness_score(author_trustworthiness_score, link_trustworthiness_score, 0.3, 0.7)
    
    # Precise boundary handling
    if avg <= 20:
        stars = 1
    elif avg <= 40:
        stars = 2
    elif avg <= 60:
        stars = 3
    elif avg <= 80:
        stars = 4
    else:
        stars = 5

    print(f"Final trustworthiness score: {avg:.2f}%")
    print(f"Star rating: {stars} stars")

    return stars


if __name__ == "__main__":
    main()