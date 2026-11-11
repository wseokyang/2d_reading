"""Legacy launch filename for the maintained Data Matrix Streamlit UI.

Run ``streamlit run app.py`` normally. ``streamlit run org_app.py`` starts the
same application. The original supplied UI is preserved in reference/org_app.py.
"""

from app import main


if __name__ == "__main__":
    main()
