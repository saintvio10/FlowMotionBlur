// MetalBackend.h - Metal-Rechenkerne fuer Premieres GPU-Weg (nur macOS).
//
// Premiere uebergibt im GPU-Weg die Bilder als Metal-Puffer (BGRA, 32 Bit Float
// oder 16 Bit Half). Dieses Modul berechnet direkt darauf:
//   * verkleinerte Luminanz (fuer die Bewegungsanalyse auf der CPU)
//   * die Bewegungsunschaerfe (Modus "Natuerlich") aus den Vektorfeldern
// Die Schnittstelle kommt ohne Objective-C-Typen aus: Metal-Objekte werden als
// void* durchgereicht (id<MTLDevice>, id<MTLCommandQueue>, id<MTLBuffer>).

#pragma once

#include "../core/FlowCore.h"

#include <string>

namespace fmb { namespace metal {

struct Frame {
    void* buffer = nullptr;   // id<MTLBuffer>
    int   w = 0, h = 0;
    int   rowBytes = 0;
    bool  isHalf = false;     // true: 4 x half (8 Byte/Pixel), sonst 4 x float
};

// Verkleinerte Luminanz (Faktor 'factor', wie LumaFromRGBA) und mittlerer Alphawert.
bool Luma(void* device, void* queue, const Frame& in, int factor, float alphaWeight,
          Image& luma, double& alphaMean);

// Bewegungsunschaerfe "Natuerlich" von src nach dst (gleiche Groesse/Format).
// fwd/bwd wie bei ApplyMotionBlur (Fluss-Aufloesung = Bild / flowScale).
// Ohne Vektorfelder wird src nach dst kopiert.
bool BlurNatural(void* device, void* queue, const Frame& src, const Frame& dst,
                 const Flow* fwd, const Flow* bwd, int flowScale, const BlurParams& p);

// Wie BlurNatural, aber direkt im Bild 'frame' (Premiere rendert im GPU-Weg
// in das Ausgabebild, das schon das aktuelle Bild enthaelt).
bool BlurNaturalInPlace(void* device, void* queue, const Frame& frame,
                        const Flow* fwd, const Flow* bwd, int flowScale, const BlurParams& p);

// Kopiert src nach dst.
bool Copy(void* device, void* queue, const Frame& src, const Frame& dst);

// Letzte Fehlermeldung (leer, wenn keine).
std::string LastError();

}} // namespace fmb::metal
