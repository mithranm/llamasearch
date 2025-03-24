import os
import gradio as gr
from llamasearch.ui.app import (
    generate_response,
    listen_answer,
    submit_review,
    save_and_clear,
    new_chat,
    create_app,
)

def test_generate_response():
    link = "https://example.com"
    user_input = "What is AI?"
    expected = "AI response for: What is AI? (Link: https://example.com)"
    assert generate_response(link, user_input) == expected

def test_listen_answer_empty():
    result = listen_answer("")
    assert result is None

def test_listen_answer_valid():
    result = listen_answer("Hello, this is a test")
    assert isinstance(result, str)
    # Verify the temporary audio file exists
    assert os.path.exists(result)
    # Clean up the temporary file
    os.remove(result)

def test_submit_review():
    review_text = "Great service!"
    expected = "Review submitted: Great service!"
    assert submit_review(review_text) == expected

def test_save_and_clear():
    # Use "👍" so rating_int should become 1
    outputs = save_and_clear("Q", "A", "👍")
    # Expect 4 updates: for user_input, chat_output, rating, and audio_output.
    assert isinstance(outputs, tuple)
    assert len(outputs) == 4
    # First three updates should clear their value
    for update in outputs[:3]:
        assert update.get("value") == ""
    # Fourth update should hide the audio output (visible set to False)
    assert outputs[3].get("visible") is False

def test_new_chat():
    outputs = new_chat()
    # Expect 7 updates: for link_input, user_input, chat_output, rating,
    # review_prompt, submit_review_btn, and audio_output.
    assert isinstance(outputs, tuple)
    assert len(outputs) == 7
    # First four updates should clear the value
    for update in outputs[:4]:
        assert update.get("value") == ""
    # Fifth update: should clear value and hide review_prompt
    assert outputs[4].get("value") == ""
    assert outputs[4].get("visible") is False
    # Sixth and seventh updates should hide their components
    for update in outputs[5:]:
        assert update.get("visible") is False

def test_app_creation():
    app = create_app()
    assert isinstance(app, gr.Blocks)
    # Gather labels from all components that have one
    components = [comp for comp in app.blocks.values() if hasattr(comp, "label")]
    labels = [comp.label for comp in components if comp.label]
    # Update expected labels to those that are actually present
    essential_labels = ["Enter a Website Link", "Ask a Question", "AI Response", "Rate Response", "Audio Response"]
    for label in essential_labels:
        assert label in labels
