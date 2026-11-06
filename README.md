# Flow Motion Blur – Plugin für Premiere Pro (und After Effects)

**Version 1.5**

Automatische, vektorbasierte Bewegungsunschärfe: Das Plugin verfolgt jeden Pixel
vom vorherigen zum nächsten Frame (optischer Fluss) und verwischt das Bild entlang
dieser Bewegung. Einfach auf einen Clip ziehen – fertig.

## Funktionen (v1.1)

| Funktion | Status |
|---|---|
| Automatisches Tracking jedes Pixels (vor- und rückwärts) | ✅ |
| Einstellbare Unschärfe (Blur Amount = Verschlusszeit, 0.5 ≙ 180°) | ✅ |
| Weiche Kanten: bewegter Vordergrund verschmiert über den Hintergrund | ✅ |
| RGB+A-Tracking (Alphakanal hilft bei dunklem/gekeytem Material) | ✅ |
| **Eigene Bewegungsvektoren** aus einer Ebene (z. B. aus 3D-Programmen) | ✅ |
| **Track Points** (6 Punkte) zur Lenkung des Trackings | ✅ |
| **GPU-Beschleunigung** (OpenCL: Nvidia, AMD, Intel, Apple) mit CPU-Fallback | ✅ |
| Vektor-Ansicht zur Kontrolle | ✅ |
| 8 Bit und 32 Bit Float in Premiere, 8/16 Bit in After Effects | ✅ |

## Parameter

| Parameter | Bedeutung |
|---|---|
| **Blur Amount (Shutter)** | Stärke. 0.5 = Film-Look (180°), 1.0 = doppelt so lang |
| **Bewegungsart** | **Kamera (Drohne, Schwenk, Fahrt)**: Die Kamerabewegung wird direkt aus den Bildern als ein perspektivisches Modell bestimmt (auch bei Speed Ramps mit sehr großer Bewegung). Je Bildstelle wird gewählt, ob Modell, feines Tracking oder Ruhe (z. B. Himmel) am besten passt. **Frei (bewegte Objekte)**: für bewegte Dinge bei ruhiger Kamera. |
| **Unschärfe-Modus** | **Natürlich (Kamera)**: jeder Pixel entlang seiner eigenen Bewegung, wie eine echte Belichtung. **Objekte**: schnelle Objekte verschmieren über den Hintergrund. |
| **Max Samples** | Qualität der Verwischung. Höher = glatter, langsamer |
| **Analyse-Qualität** | Auflösung der Bewegungsanalyse (Schnell ¼, Normal ½, Hoch voll) |
| **Vektor-Glättung** | Glättet das Bewegungsfeld; höher bei unruhigen Ergebnissen |
| **RGB+A-Tracking** | Nutzt den Alphakanal für die Analyse mit |
| **GPU verwenden** | Rechnet per OpenCL auf der Grafikkarte, sonst automatisch CPU |
| **Ansicht** | „Bewegungsvektoren“ zeigt Richtung (Farbe) und Stärke (Helligkeit); „Diagnose“ schreibt Pixelformat, Zeitschritte, Status der Nachbarframes und des Kameramodells ins Bild |
| **Vektor-Quelle** | „Automatisch (Tracking)“ oder „Vektor-Ebene“ |

### Eigene Bewegungsvektoren

1. Vektor-Render (z. B. aus Blender, Cinema 4D, Maya) als Clip auf eine eigene Spur legen
   (Spur ausblenden, damit sie nicht sichtbar ist).
2. **Vektor-Quelle** = „Vektor-Ebene“, unter **Eigene Bewegungsvektoren → Vektor-Ebene** die Spur wählen.
3. Format einstellen:
   - **Rot** = Bewegung in X, **Grün** = Bewegung in Y, in Pixeln pro Frame (volle Auflösung).
   - **Normalisiert**: 0.5 = keine Bewegung, 0 bzw. 1 = ± „Max. Verschiebung“.
   - **Pixel (32-Bit Float)**: Werte sind direkt Pixel (z. B. OpenEXR-Vektorpässe).
   - **Y umkehren**, falls das 3D-Programm Y nach oben zählt (Bewegung wirkt sonst gespiegelt).
4. Tipp: Normalisierte Vektoren als 16 Bit oder Float exportieren. In 8 Bit lässt sich
   „keine Bewegung“ (0.5) nicht exakt speichern; das ergibt minimale Restbewegung.

Ist keine Ebene gewählt, wird automatisch getrackt.

### Track Points

Wenn das Tracking eine Stelle falsch erkennt (sehr schnelle Bewegung, Verdeckung,
wenig Struktur):

1. Unter **Track Points** einen Punkt aktivieren („Punkt 1 aktiv“).
2. Den Punkt im Programmmonitor auf ein markantes Detail setzen und über den Clip
   **per Keyframes mitführen** (er muss sich mitbewegen, wie ein Tracker).
3. **Einflussradius** bestimmt, wie weit um den Punkt herum seine Bewegung gilt.

Das Plugin liest die Punktposition im aktuellen, vorherigen und nächsten Frame und
erzwingt dort genau diese Bewegung. Ein Punkt ohne Keyframes bedeutet „hier bewegt
sich nichts“.

### GPU

- Nutzt OpenCL, das bei Nvidia-, AMD- und Intel-Treibern unter Windows sowie unter
  macOS mitgeliefert wird. Nichts extra installieren.
- Ohne passende GPU oder bei einem Fehler rechnet das Plugin automatisch auf der CPU.
  GPU und CPU liefern identische Ergebnisse.
- „Über …“ im Effekt zeigt, welche GPU erkannt wurde.
- Hinweis: Premiere zeigt den Effekt nicht mit dem „GPU-beschleunigt“-Symbol an, weil
  dieses nur für Effekte gilt, die Premieres eigene GPU-Schnittstelle nutzen. Die
  Berechnung läuft trotzdem auf der Grafikkarte.
- Zur Fehlersuche: Umgebungsvariable `FMB_GPU_DEBUG=1` gibt OpenCL-Fehler aus,
  `FMB_DISABLE_GPU=1` schaltet die GPU global ab.

## Bauen

Benötigt:
- **After Effects SDK** (kostenlos: https://developer.adobe.com/after-effects/ → „Download SDK“).
  Premiere Pro lädt Effekte, die mit diesem SDK gebaut sind, direkt.
- **CMake** ≥ 3.20
- **Windows:** Visual Studio 2022 (C++-Desktopentwicklung)
- **macOS:** Xcode (Command Line Tools)

Für die GPU wird beim Bauen nichts zusätzlich benötigt (OpenCL wird zur Laufzeit geladen).

### Windows

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DAE_SDK="C:/Pfad/zum/AfterEffectsSDK/Examples"
cmake --build build --config Release
```

Ergebnis: `build\Release\FlowMotionBlur.aex`

### macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAE_SDK="/Pfad/zum/AfterEffectsSDK/Examples"
cmake --build build
```

Ergebnis: `build/FlowMotionBlur.plugin`

## Installieren

Datei in den gemeinsamen Adobe-Plugin-Ordner kopieren (gilt für Premiere **und** After Effects):

- **Windows:** `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\`
- **macOS:** `/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/`

Premiere neu starten → Effekte → **Flow Motion Blur** → auf den Clip ziehen.

## Testen ohne Adobe

Der Rechenkern ist vom SDK getrennt und lässt sich überall testen:

```bash
cmake -S . -B build && cmake --build build && ./build/fmb_test .
```

Der Test misst die Tracking-Genauigkeit, prüft die Track Points, vergleicht GPU und
CPU (falls OpenCL vorhanden) und schreibt `original.ppm`, `blur_q*.ppm`, `vectors.ppm`.

## Aufbau

```
src/core/FlowCore.*              Optischer Fluss, Track Points, Bewegungsunschärfe (CPU)
src/gpu/GpuBackend.*             Dieselben Schritte per OpenCL
src/gpu/Kernels.h                OpenCL-Kernel (zur Laufzeit kompiliert)
src/gpu/ClApi.h                  OpenCL-Funktionen, dynamisch geladen
src/plugin/FlowMotionBlur.*      Anbindung an Premiere/After Effects
src/plugin/FlowMotionBlurPiPL.r  Plugin-Beschreibung für den Host
tests/                           Tests (Kern, GPU-Vergleich, OpenCL-Deklarationen)
```

## Bekannte Einschränkungen

- Am ersten/letzten Frame eines Clips wird nur in eine Richtung getrackt.
- Wie bei jedem Optical-Flow-Verfahren können bei Verdeckungen, Lichtwechseln
  oder sehr schnellen Bewegungen Artefakte entstehen → Track Points helfen.
- Vektor-Ebenen werden mit voller Auflösung des Clips erwartet.
