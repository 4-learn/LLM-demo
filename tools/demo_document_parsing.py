"""Generate a tiny deterministic PDF fixture for lesson 20 (standard library only).

The fixture has selectable ASCII text. It is not a PDF parser, layout-table,
scanned-image, or OCR sample. All offsets and stream lengths are self-contained.
"""
from pathlib import Path


def make_pdf(path):
    stream = (b"BT /F1 16 Tf 50 780 Td (Work order WO-2048) Tj "
              b"0 -30 Td (Asset: Pump A) Tj 0 -24 Td (Status: Complete) Tj ET\n")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
    ]
    data, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n".encode())
    Path(path).write_bytes(data)
    return len(data)


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo-work-order.pdf")
    print(f"PDF fixture: {target} ({make_pdf(target)} bytes)")
