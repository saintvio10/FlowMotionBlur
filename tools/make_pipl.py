#!/usr/bin/env python3
"""Erzeugt die binaere PiPL-Ressource fuer Windows (ersetzt Adobes PiPLtool.exe).

Format (Windows, little endian):
  int16 1, int32 version(0), int32 count,
  je Property: vendor(4) key(4) int32 id(0) int32 length, Daten (auf 4 Byte aufgefuellt)
Vier-Zeichen-Codes werden als int32 gespeichert, liegen also byte-umgekehrt im Speicher.
"""
import struct, sys

def code(s):            # 'kind' -> int32 LE (Bytes "dnik")
    return struct.pack('<I', int.from_bytes(s.encode('ascii'), 'big'))

def pstring(s):
    b = s.encode('latin-1')
    return bytes([len(b)]) + b

def prop(key, data, declared=None):
    # Laenge = aufgefuellte Datenlaenge; nur eINF traegt (wie bei PiPLtool) die echte Laenge 2.
    pad = (-len(data)) % 4
    length = declared if declared is not None else len(data) + pad
    return code('8BIM') + code(key) + struct.pack('<ii', 0, length) + data + b'\0' * pad

def pipl(name, category, match, entry, spec_ver, spec_sub, version, outflags, outflags2):
    props = [
        prop('kind', code('eFKT')),
        prop('name', pstring(name)),
        prop('catg', pstring(category)),
        prop('8664', entry.encode('ascii') + b'\0'),
        prop('ePVR', struct.pack('<hh', 2, 0)),
        prop('eSVR', struct.pack('<hh', spec_ver, spec_sub)),
        prop('eVER', struct.pack('<I', version)),
        prop('eINF', struct.pack('<h', 0), declared=2),
        prop('eGLO', struct.pack('<I', outflags)),
        prop('eGL2', struct.pack('<I', outflags2)),
        prop('eMNA', pstring(match)),
        prop('aeFL', struct.pack('<I', 0)),
    ]
    return struct.pack('<hii', 1, 0, len(props)) + b''.join(props)

if __name__ == '__main__':
    out = sys.argv[1]
    data = pipl('Flow Motion Blur', 'Flow Motion Blur', 'FMB FlowMotionBlur', 'EffectMain',
                13, 29, 1114113, 0x02000006, 0)
    open(out, 'wb').write(data)
    print(f'{out}: {len(data)} Bytes')
