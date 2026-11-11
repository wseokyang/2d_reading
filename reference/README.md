# Supplied implementations

These three files are byte-identical copies of the implementations supplied
before consolidation. `SHA256SUMS` records their original hashes. They are
reference material; the active application does not import them.

The maintained Streamlit application is `../app.py`. The root `org_app.py`
launches that application, `org_code_2.py` forwards the public reader API, and
`org_code.py` adapts the original default BGR read call to the Data Matrix reader.

Run `sha256sum -c SHA256SUMS` from this directory to verify the originals.
