# tests/test_main.py
"""Tests for main module."""
from llamasearch.main import main

def test_main(capsys):
    """Test main function outputs expected greeting."""
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello from LlamaSearch!\n"