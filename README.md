# LlamaSearch

RAG-based search application developed for CS 321 at George Mason University.

## Team

- Leo Angulo (langulo@gmu.edu)
- Georgi Zahariev (gzaharie@gmu.edu)
- Robin Hwang (rhwang4@gmu.edu)
- Mithran Mohanraj (mithran.mohanraj@gmail.com)
- Dhanush Reddy Kandukuri (kandukuri.dhanush@gmail.com)
- Shaheda Tawakalyar (stawakal@gmu.edu)
- Malek Bashagha (abashagh@gmu.edu)

## Development Setup

### Prerequisites

- Python 3.12 or higher
- Conda package manager (or environment manager of your choice)

### Environment Setup

1. Create and activate the conda environment:
```bash
conda create -n cs321 python=3.12
conda activate cs321
```

2. Install the package in development mode:
```bash
pip install -e .
```

### Verifying Installation

Run these commands from the project root to verify everything is working:

```bash
# Run tests with coverage report
pytest

# Check code style
flake8

# Run the application
python -m llamasearch.main
```

## Development Workflow

0. Ensure you are working on an issue branch
1. Activate the conda environment:
```bash
conda activate cs321
```

2. Make your changes
3. Run tests and linting before committing:
```bash
pytest
flake8
```

4. Create a pull request to merge your issue branch

## CI/CD

The project uses GitHub Actions for continuous integration, which automatically:
- Runs all tests
- Checks code style with flake8
- Generates test coverage reports