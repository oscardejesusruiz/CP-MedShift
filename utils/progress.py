"""
Progress printing of the long notebook sweeps.

- status:  overwrite the current line ('\r'), e.g. 'CheXpert  KL = 0.5  draw 3/20  ·  FCA'
- hms:     elapsed seconds as '01h 02m 03s'
- chunked: apply fn to x in chunks, printing '<tag>  ·  <what> i/M' after every chunk
"""

import numpy as np


def status(msg, width=125):
    """Print msg over the current line (no newline)."""
    print('\r' + msg.ljust(width), end='', flush=True)


def hms(sec):
    """Seconds as 'HHh MMm SSs'."""
    sec = int(sec)
    return f'{sec // 3600:02d}h {sec % 3600 // 60:02d}m {sec % 60:02d}s'


def chunked(fn, x, tag, what, chunk=2000, **kw):
    """fn(x[b0:b1], **kw) over chunks of `chunk` rows of x, concatenated. Keyword arrays with len(x) rows are
    sliced like x (e.g. per-image kernel rows); the others are passed unchanged."""
    M, out = len(x), []
    for b0 in range(0, M, chunk):
        b1 = min(b0 + chunk, M)
        status(f'{tag}  ·  {what} {b1}/{M}')
        out.append(fn(x[b0:b1], **{k: (v[b0:b1] if isinstance(v, np.ndarray) and len(v) == M else v) for k, v in kw.items()}))
    return np.concatenate(out)
