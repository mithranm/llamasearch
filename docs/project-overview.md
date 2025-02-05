# Project Overview - LLAMASEARCH

LLAMASEARCH is an advanced search tool with a function of delivering relevant information and answer for a website link entered in a query by a user. Users can enter a query, and then an app will use Llama LLM to generate an answer that comes from information in a website.

These are some of the features that are offered by LLamaSearch:

1. Text-based output generated through the model
2. A quotations and excerpts file with information that addresses the question
3. An audio record of the generated answer
4. A 1 to 10 trust and legitimacy rating system for measuring website/sourcing
5. Pop-up feedback feature and request for feedback

## Key Features

LLAMASEARCH isn’t a mere search tool but one designed with usability, accuracy, and privacy in consideration. Here are some more of the features that distinguishes this from others:

1. Privacy-First Search – Unlike most search tools powered by AI, which collect and use your information, LLAMASEARCH stores nothing and will not use your queries to build any AI model.

2. Targeted Website Queries – Unlike a flood of irrelevant information from the entire web, you can target a specific website to search in. That is, specific, relevant, and contextual information.

3. Smart Summarization – There is no necessity to wade through lengthy articles and crowded pages. LLAMASEARCH brings concise, well-written summaries specific to your search, conserving your time and efforts.  

4. Audio Playback for Answers – Do not enjoy reading? Have your summaries heard aloud with our inbuilt audio feature, for easier information absorption when in motion.

5. Citations and source highlighting – Transparency is key. All answer is accompanied with citations, clearly stating sources for information and highlighting important sections of the source.

6. Downloadable Search History – Want to review an older search? You can download and save your search history for future use—great for studying and researching.

7. User Feedback Facility – Your feedback matters to us! Users can leave feedback and make recommendations, allowing us to make improvements to the experience.

8. Website Credibility Scores – Not all web sources are alike. With our application, you can assess your sources' credibility through our website reliability ratings, gauging them in terms of content value, level of bias, and accuracy of factuality.

With all these features in one tool, LLAMASEARCH isn’t a search tool, but a wiser, privacy-aware research assistant that you can integrate in your daily life.

## Tech Stack  

Here is the Tech Stack utilized in the development of LLamaSearch:

Frontend (User Interface)
1. Gradio – Gives an easy web platform for its ease of access in searching with its search engine.

Backend (Server & Processing)
1. Python – Python, a backend language, assists in communicating with AI models with ease, creating APIs, and processing information with high efficiency.
2. Llama (Meta AI Model) – Used for high-quality summaries via natural language processing.
3. Jina API - High-performance and AI-native neural search platform with expandable capabilities for efficient processing and retrieval of information in unstructured and structured information.

AI & NLP (Search Intelligence)
1. Meta’s Llama Model – AI model for processing all queries and generating relevant and concise output with regard to queries received.
2. Hugging Face Transformers – Enables processing of text and deepens query understanding.
3. Speech Synthesis API – Converts text into speech for listening for deaf and speech-impaired visitors who prefer listening over reading.

Database & Storage
1. ChromaDB – Efficient high-performance vector database for storing and searching for embeddings with increased search efficiency and accuracy.
2. SQlite –  Relational Database that deals with metadata and structures information in a most efficient manner.

With this technology stack, LLAMASEARCH will provide a private, efficient, and smart search for its users.

# Installation Instructions

Setting up LLAMASEARCH is very straightforwad. Follow these steps to install the required dependencies and get the project running on your machine.

**Prerequisites**

Before you begin, make sure you have the following installed:
1. Python 3.12 or higher – Required for running the backend.
2. Conda package manager – Used for managing dependencies and virtual environments.
3. Git – To clone the repository and contribute to development.

**Environment Setup**
1. Clone the Repository 
   First, grab a local copy of the project by running:
   ```sh
   git clone https://github.com/your-repo/llamasearch.git
   cd llamasearch

**Create and Activate the Conda Environment**

Set up an isolated Python environment for the project:

```sh
conda create -n cs321 python=3.12
conda activate cs321
```

**Install the Project Dependencies**

Install all required packages in development mode:

```sh
pip install -e .
```

**Verifying Installation**

To confirm everything is set up correctly, run these checks from the project root:

1. Run tests with a coverage report:
    ```sh
    pytest
    ```
2. Check code style compliance:
    ```sh
    flake8
    ```
3. Run the application
    ```sh
    python -m llamasearch.main
    ```

If all checks pass, you’re good to go to move on.

**Development Workflow**

For contributors, follow these steps:

1. Ensure you’re working on a separate issue branch:
   ```sh
   git checkout -b feature-branch
   ```
2. Activate the Conda environment:
   ```sh
   conda activate cs321
   ```
3. Make your code changes and test them:
   ```sh
   pytest
   flake8
   ```
4. Commit your changes and create a pull request:
   ```sh
   git add .
   git commit -m "Description of changes"
   git push origin feature-branch
   ```
5. Submit a pull request for review and merge.




See [ADR-001](./ADRs/001-initial-architecture.md) for more details.