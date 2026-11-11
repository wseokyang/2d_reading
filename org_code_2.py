"""Original reader import path, backed by the maintained reader implementation.

The public reader API accepts PIL images or encoded image bytes. OpenCV BGR
arrays belong to backend.decode_datamatrix_from_array instead. The supplied
implementation is preserved in reference/org_code_2.py.
"""

from reader import CodeHit, ReaderOptions, ReadResult, read_datamatrix, read_image_bytes

__all__ = ["CodeHit", "ReaderOptions", "ReadResult", "read_datamatrix", "read_image_bytes"]
