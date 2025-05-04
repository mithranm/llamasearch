# LlamaSearch

LlamaSearch is a modular tool for crawling, indexing, and querying documents. It can be used via a Graphical User Interface (GUI), a Model Context Protocol (MCP) server, or a Command-Line Interface (CLI). The storage directories for crawl data, indices, logs, and models are managed dynamically at runtime so that you may easily plug and play with different data sets.

LlamaSearch embodies the GNU/FSF spirit by exposing a consolidated entry point that lets you run it in multiple modes.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Modes of Operation](#modes-of-operation)
  - [GUI Mode](#gui-mode)
  - [MCP Server Mode](#mcp-server-mode)
  - [CLI Mode](#cli-mode)
    - [Crawl](#crawl)
    - [Index Local Files](#index-local-files)
    - [Search](#search)
    - [Export Data](#export-data)
    - [Set Storage Directory](#set-storage-directory)
- [Example Workflow](#example-workflow)
- [Future Enhancements](#future-enhancements)

## Installation

1. **Clone the repository** (if using source distribution):

   ```bash
   git clone https://github.com/yourusername/llamasearch.git
   cd llamasearch
   ```

2. **Install dependencies**:

   It is recommended to create a virtual environment and then install required packages:
   
   ```bash
   python -m venv env
   source env/bin/activate  # on Linux/MacOS, or `env\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. **Model Files**:

   Place your model (e.g. a `.gguf` file for llama-cpp or a model identifier for Hugging Face) into the default models directory (`~/.llamasearch/models` by default) or specify an alternate directory using environment variables.

## Configuration

LlamaSearch uses a dynamic storage system managed by its Data Manager. The default directories are:

- **Base Directory**: `~/.llamasearch` (you can override this by setting the `LLAMASEARCH_DATA_DIR` environment variable)
- **Crawl Data**: `~/.llamasearch/crawl_data`
- **Index**: `~/.llamasearch/index`
- **Models**: `~/.llamasearch/models`
- **Logs**: `~/.llamasearch/logs`

The user can change these settings at runtime using the CLI subcommand `set` (see below) or by editing the settings file stored at `~/.llamasearch/settings.json`. You can also export your crawl data and index to a tar.gz archive for sharing.

## Modes of Operation

LlamaSearch can be used in three primary modes:

### GUI Mode

This is the default mode. Simply run:
  
```bash
python -m llamasearch
```

The GUI provides a full-featured interface with tabs for:
- **Settings**: Change model and system parameters.
- **Search & Index**: Ask a question and specify root URLs for crawling; also add local files.
- **Terminal**: View live logs and debug output.
  
Use the minimal UI if you prefer the consolidated view.

### MCP Server Mode

To run LlamaSearch as an MCP server (for use by external applications):

```bash
python -m llamasearch --mode mcp
```

This launches a server (typically on port 8001) that exposes a standardized API (using Model Context Protocol) for external processes to:
- List available tools.
- Request a crawl and indexing.
- Perform search queries.

Make sure [uvicorn](https://www.uvicorn.org/) is installed (`pip install uvicorn`).

### CLI Mode

In CLI mode, you can run various subcommands. All CLI commands can run in a headless fashion (without the GUI). By default, if you don’t need LLM generation, the system uses a lightweight `NullModel`. You can request an AI-generated response using the `--generate` flag. In CLI mode the following subcommands are available:

#### Crawl

Start a crawl from a specified URL. In this mode, LlamaSearch will crawl the website using an asynchronous, concurrent BFS. It indexes the pages it finds (by converting content as necessary) into the index directory.

**Example:**

```bash
python -m llamasearch --mode cli crawl --url "https://example.com" --target-links 15 --max-depth 2 --phrase "python" --api-type jina
```

This command begins crawling from `https://example.com`, aiming to collect up to 15 links, with a maximum depth of 2, and prioritizes links that include the word “python.” Use `--api-type mithran` (with `--key-id` and `--private-key` if needed) if you prefer the private API.

#### Index Local Files

Index a local file or directory. LlamaSearch will automatically convert Office documents or PDFs to Markdown as needed and add them to the index.

**Example:**

```bash
python -m llamasearch --mode cli index --path "/path/to/my/documents"
```

#### Search

Query the existing index and optionally generate an AI response.

- **Without generation:** (Simply retrieves chunks from the index.)
  
  ```bash
  python -m llamasearch --mode cli search --query "What is artificial intelligence?"
  ```

- **With generation:** (Uses a real LLM such as a HuggingFace-backed model to generate an AI answer.)
  
  ```bash
  python -m llamasearch --mode cli search --query "What is artificial intelligence?" --generate
  ```

If the `--generate` flag is omitted, a lightweight `NullModel` is used, and you get only the indexed chunks.

#### Export Data

Export one or more data directories (e.g. crawl data, index) as a tar.gz archive. This makes it easy to share or back up your indexed data.

**Example:**

```bash
python -m llamasearch --mode cli export --keys crawl_data index --output export_data.tar.gz
```

This command exports the `crawl_data` and `index` directories into an archive named `export_data.tar.gz`.

#### Set Storage Directory

Change one of the storage directories at runtime. For example, to change where crawl data is stored:

**Example:**

```bash
python -m llamasearch --mode cli set --key crawl_data --path "/new/path/to/crawl_data"
```

This immediately updates the runtime configuration (and saves it to a settings file) so that future operations use the new directory.

## Example Workflow

1. **Start the GUI**:  
   Run `python -m llamasearch` to launch the graphical interface. Use the “Search & Index” tab to:
   - Enter your question,
   - Provide one or more root URLs for crawling,
   - Optionally pick local files to index,
   - Then click “Search” to trigger a crawl and search.

2. **CLI Crawling for a Specific Project**:  
   Run:  
   ```bash
   python -m llamasearch --mode cli crawl --url "https://example.com" --target-links 15 --max-depth 2 --phrase "machine learning"
   ```
   This will asynchronously crawl and index the pages from the given URL into your index.

3. **Local File Indexing**:  
   After preparing some documents in a directory, run:
   ```bash
   python -m llamasearch --mode cli index --path "/path/to/my/docs"
   ```
   The system will convert Office/PDF files to Markdown (if necessary) and add them to the index.

4. **Searching the Index**:  
   To query your indexed documents (without an AI-generated answer), run:
   ```bash
   python -m llamasearch --mode cli search --query "Explain the concept of neural networks"
   ```
   To get an LLM-generated answer, add `--generate`.

5. **Exporting Data**:  
   After a long crawl or indexing session, export your data for backup or sharing:
   ```bash
   python -m llamasearch --mode cli export --keys crawl_data index --output my_project_export.tar.gz
   ```

6. **Changing Storage Directories**:  
   If you want to maintain separate projects with different data directories, run:
   ```bash
   python -m llamasearch --mode cli set --key index --path "/different/path/for/index"
   ```
   This updates the index directory on the fly.