"""
Artistic Style Transfer - Universal Entry Point for Streamlit Cloud and Local Execution.
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import and execute the UI
import simple_ui
