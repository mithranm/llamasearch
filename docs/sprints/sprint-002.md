# Sprint 2

## File Structure

```mermaid
flowchart TD;
    A[llamasearch/] --> B[sources/]
    A --> C[trustworthiness/]
    B --> D[trustedSources.txt]
    C --> F[authorSearch.py]
    F --> G[authorCrawl.py]
    G --> H[linkChecker.py]
    H --> I[score.py]
```

## Tasks

1. Create Database

   1. Create seperate folder named sources/ under llamasearch/
   2. Find 50-100 website sources that can be considered reliable sources for a wide range of topics
   3. Store as a text file or python file whatever is the most convenient to read from

2. Creating link checker

   1. Create a file linkChecker.py that will communicate with the database
   2. linkChecker.py should recieve a list of links and be able to compare to the database to identify any relative matches.
   3. linkChecker.py is responsible for checking the links from the output of both crawler.py and authorCrawl.py
   4. linkChecker.py should be able to give a count or a boolean indicating that the associated links are found in the database.

3. Create author searching

   1. authorSearch.py should be able to look through the output file from extractor.py and to find author names.
   2. authorCrawl.py should be able to run a search from the output of authorSearch.py to find links associated with the extracted authors.
   3. authorCrawl.py should output a text file containing the links associated to the authors found.

4. Create trustworthiness score protocol 

    1. score.py should be able to interpret the data from linkChecker.py
    2. score.py needs to have a protocol determining how trustworthy the sources are from the linkChecker.py
    3. score.py needs to output a score that app.py will be able to interpret.