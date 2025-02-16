from gradio.blocks import Blocks
from llamasearch.ui.app import create_app


def test_app_creation():
    """Test that the app is created successfully with basic components"""
    app = create_app()

    # Verify app is created as Gradio Blocks instance
    assert isinstance(app, Blocks)

    # Get all components with labels
    components = [comp for comp in app.blocks.values() if hasattr(comp, "label")]
    labels = [comp.label for comp in components if comp.label]

    # Verify essential components are present
    essential_labels = ["Enter a Website Link", "AI Response", "Ask a Question"]

    for label in essential_labels:
        assert label in labels
