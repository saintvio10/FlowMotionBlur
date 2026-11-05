#!/usr/bin/env bash
# Baut FlowMotionBlur.aex fuer Windows x64 auf Linux (MinGW-w64), ohne Visual Studio.
# Aufruf: tools/build_windows_mingw.sh /pfad/zum/AE_SDK/Examples   (Ordner mit Headers/ und Util/)
set -euo pipefail
SDK="${1:?Pfad zum SDK-Ordner mit Headers/ und Util/ angeben}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build-win"
mkdir -p "$OUT"
python3 "$ROOT/tools/make_pipl.py" "$OUT/pipl.bin"
( cd "$OUT" && echo '16000 PiPL "pipl.bin"' > pipl.rc && x86_64-w64-mingw32-windres pipl.rc -O coff -o pipl.o )
x86_64-w64-mingw32-g++-posix -O2 -std=c++17 -shared -Wno-multichar \
  -DMSWindows -D_WINDOWS -DWIN32 -DWIN_ENV -DNDEBUG \
  -I"$SDK/Headers" -I"$SDK/Headers/SP" -I"$SDK/Util" \
  "$ROOT/src/plugin/FlowMotionBlur.cpp" "$ROOT/src/core/FlowCore.cpp" "$ROOT/src/gpu/GpuBackend.cpp" "$OUT/pipl.o" \
  -static -static-libgcc -static-libstdc++ -s -o "$OUT/FlowMotionBlur.aex"
echo "Fertig: $OUT/FlowMotionBlur.aex"
