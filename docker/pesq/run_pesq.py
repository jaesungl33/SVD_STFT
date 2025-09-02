import sys, json
from pathlib import Path
import soundfile as sf
import numpy as np
from pesq import pesq as pesq_wb

SR = 16000

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: run_pesq.py REF.wav DEG.wav"}))
        sys.exit(1)
    ref_p = Path(sys.argv[1])
    deg_p = Path(sys.argv[2])
    if not ref_p.exists() or not deg_p.exists():
        print(json.dumps({"error": "missing file"}))
        sys.exit(1)
    ref, sr1 = sf.read(str(ref_p))
    deg, sr2 = sf.read(str(deg_p))
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    if deg.ndim > 1:
        deg = deg.mean(axis=1)
    if sr1 != SR or sr2 != SR:
        print(json.dumps({"error": "require 16kHz wav"}))
        sys.exit(1)
    try:
        mos = float(pesq_wb(SR, ref.astype(np.float32), deg.astype(np.float32), 'wb'))
        print(json.dumps({"pesq_wb": mos}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()
