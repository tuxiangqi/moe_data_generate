"""CPU checks for buffered MoE capture publication."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from data_generate.capture_writer import BufferedCaptureWriter, FILES, READY, wait_for_capture


def rows(*values):
    return tuple(np.asarray(values, dtype=dtype)[:, None]
                 for dtype in (np.int32, np.float32, np.float32, np.uint16))


class CaptureWriterTests(unittest.TestCase):
    def test_interleaved_requests_are_written_in_order_before_ready(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = BufferedCaptureWriter(root, "{}")
            try:
                writer.submit({"a": rows(1, 2), "b": rows(10)})
                writer.submit({"a": rows(3), "b": rows(11, 12)})
                writer.finish(["a"])
                wait_for_capture(root / "a", timeout=2)
                self.assertFalse((root / "b" / READY).exists())
                writer.submit({"b": rows(13)})
                writer.finish(["b"])
                wait_for_capture(root / "b", timeout=2)
                for name, dtype in zip(FILES, (np.int32, np.float32, np.float32, np.uint16),
                                       strict=True):
                    np.testing.assert_array_equal(np.fromfile(root / "a" / name, dtype), [1, 2, 3])
                    np.testing.assert_array_equal(np.fromfile(root / "b" / name, dtype),
                                                  [10, 11, 12, 13])
            finally:
                writer.close()

    def test_writer_error_reaches_reader_without_marking_request_ready(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = BufferedCaptureWriter(root, "{}")
            (root / "blocked").write_text("not a request directory")
            writer.submit({"blocked": rows(1)})
            writer.finish(["blocked"])
            try:
                with self.assertRaisesRegex(RuntimeError, "capture writer failed"):
                    wait_for_capture(root / "blocked", timeout=2)
            finally:
                with self.assertRaises(FileExistsError):
                    writer.close()


if __name__ == "__main__":
    unittest.main()
