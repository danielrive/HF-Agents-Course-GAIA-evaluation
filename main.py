"""
Main entry point for Hugging Face Space deployment.
This file imports and runs the GAIA agent from the app/gaia/runner module.
"""

from app.gaia.runner import demo

if __name__ == "__main__":
    demo.launch(debug=False, share=False)
