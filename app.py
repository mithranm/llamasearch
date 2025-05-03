# src/llamasearch/ui/app.py
import streamlit as st
from llamasearch.ui.app_logic import LlamaSearchApp
from llamasearch.ui.components import header_component
from llamasearch.ui.views.crawl_view import CrawlView
from llamasearch.ui.views.search_view import SearchView
from llamasearch.ui.views.settings_view import settings_view

def main():
    st.set_page_config(page_title="LlamaSearch", layout="wide")
    # Initialize the backend application instance
    app_instance = LlamaSearchApp(use_cpu=False, debug=False)
    
    # Render header
    header_component()
    
    # Create tabs for each section
    tab1, tab2, tab3 = st.tabs(["Crawl Website", "Search Content", "Settings"])
    
    with tab1:
        CrawlView(app_instance)
    
    with tab2:
        SearchView(app_instance)
    
    with tab3:
        settings_view(app_instance)

if __name__ == "__main__":
    main()
