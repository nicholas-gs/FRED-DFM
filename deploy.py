"""Main entry point for streamlit.
Since Streamlit does not support running modules:
    https://github.com/streamlit/streamlit/issues/662.

Run 'streamlit run deploy.py' at root directory of this project.
"""

import runpy

runpy.run_module("app.main", run_name="__main__", alter_sys=True)
