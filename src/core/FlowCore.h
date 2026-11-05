// FlowCore.h - Host-unabhaengiger Kern von "Flow Motion Blur".
//
// Enthaelt:
//   * dichten optischen Fluss (pyramidales Lucas-Kanade mit Warping)
//   * Lenkung des Flusses durch Track Points
//   * vektorbasierte Bewegungsunschaerfe (Rekonstruktionsfilter)
//   * Visualisierung der Bewegungsvektoren
//
// Der Kern kennt weder das After-Effects- noch das Premiere-SDK und kann
// daher auch ohne Adobe-Software kompiliert und getestet werden
// (siehe tests/). Eine GPU-Variante derselben Schritte steckt in src/gpu/.

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace fmb {

// Einkanaliges Float-Bild (z. B. Luminanz).
struct Image {
    int w = 0, h = 0;
    std::vector<float> d;

    void resize(int width, int height, float value = 0.f) {
        w = width; h = height;
        d.assign(static_cast<size_t>(w) * h, value);
    }
    float*       row(int y)       { return d.data() + static_cast<size_t>(y) * w; }
    const float* row(int y) const { return d.data() + static_cast<size_t>(y) * w; }
    bool empty() const { return w <= 0 || h <= 0; }
};

// RGBA-Bild, interleaved, vormultipliziert, 1.0 = weiss / volle Deckkraft.
struct RGBAImage {
    int w = 0, h = 0;
    std::vector<float> d; // r,g,b,a

    void resize(int width, int height) {
        w = width; h = height;
        d.assign(static_cast<size_t>(w) * h * 4, 0.f);
    }
    float*       px(int x, int y)       { return d.data() + (static_cast<size_t>(y) * w + x) * 4; }
    const float* px(int x, int y) const { return d.data() + (static_cast<size_t>(y) * w + x) * 4; }
};

// Dichtes Vektorfeld. (u,v) = Verschiebung in Pixeln der Fluss-Aufloesung.
struct Flow {
    Image u, v;
    int w() const { return u.w; }
    int h() const { return u.h; }
    bool empty() const { return u.empty(); }
};

// Track Point: an Position (x,y) bewegt sich das Bild um (dx,dy).
// Angaben in Pixeln der Bilder, die an ComputeFlow uebergeben werden.
struct Guide {
    float x = 0, y = 0, dx = 0, dy = 0;
};

struct FlowParams {
    int   maxLevels      = 6;      // Pyramidenstufen (begrenzt die max. Bewegung)
    int   iterations     = 4;      // Warping-Iterationen pro Stufe
    int   windowRadius   = 3;      // Lucas-Kanade-Fenster (Radius in px)
    float regularization = 2e-4f;  // Stabilisiert flaechige Bereiche
    int   smoothRadius   = 2;      // Glaettung des Endergebnisses
    float guideRadius    = 40.f;   // Einflussradius der Track Points (px, Fluss-Aufloesung)
    float initU          = 0.f;    // Startwert der Bewegung (globale Verschiebung, px von I0)
    float initV          = 0.f;
};

// Art der Unschaerfe
enum BlurMode {
    kBlurNatural = 0, // jeder Pixel entlang seiner eigenen Bewegung (wie eine echte Kamera)
    kBlurObjects = 1  // Rekonstruktionsfilter: schnelle Objekte verschmieren ueber den Hintergrund
};

struct BlurParams {
    float shutter    = 0.5f; // 0.5 = 180 Grad Verschluss, 1.0 = 360 Grad
    int   maxSamples = 32;   // Obergrenze Samples pro Pixel
    int   mode       = kBlurNatural;
};

// Feste Konstanten des Rekonstruktionsfilters (CPU und GPU identisch).
constexpr float kMaxBlurRadius = 256.f; // max. Unschaerfe je Richtung in px
constexpr float kSoftZ         = 1.0f;  // weicher Uebergang Vorder-/Hintergrund

// Fuehrt fn(begin, end) parallel in Teilbereichen aus.
void ParallelFor(int begin, int end, const std::function<void(int, int)>& fn);

// Verkleinert ein Bild um einen ganzzahligen Faktor (Box-Mittelung).
void Downscale(const Image& src, int factor, Image& dst);

// Luminanz aus RGBA, gleichzeitig um 'factor' verkleinert.
// alphaWeight > 0 mischt den Alphakanal ein ("RGB+A-Tracking"), hilfreich bei
// dunklem, maskiertem oder gekeytem Material.
void LumaFromRGBA(const RGBAImage& src, int factor, Image& dst, float alphaWeight = 0.f);

// Mischt die Track Points in ein Vektorfeld ein. 'scale' rechnet die
// Guide-Koordinaten in die Aufloesung des Felds um.
void ApplyGuides(Flow& f, const std::vector<Guide>& guides, float radius, float scale);

// Grobe globale Verschiebung von I0 nach I1 (Suche auf stark verkleinertem Bild).
// Dient als Startwert fuer ComputeFlow, damit auch sehr grosse Bewegungen
// (Speed Ramps) gefunden werden. Ergebnis in Pixeln von I0.
void EstimateGlobalShift(const Image& I0, const Image& I1, float& dx, float& dy);

// Kameramodell: Bei Kamerabewegung (Drohne, Schwenk, Fahrt) folgt die Bewegung
// im ganzen Bild einer perspektivischen Abbildung (Homographie). Diese wird
// robust (RANSAC) aus den verlaesslichen Vektoren geschaetzt und ersetzt das
// Vektorfeld. Gibt false zurueck, wenn kein Modell passt (dann unveraendert).
bool FitCameraModel(const Image& I0, Flow* fwd, Flow* bwd);

// Dichter Fluss von I0 nach I1: I0(x) ~ I1(x + flow(x)).
void ComputeFlow(const Image& I0, const Image& I1, const FlowParams& p, Flow& out,
                 const std::vector<Guide>* guides = nullptr);

// Verbessert berechnete Vektorfelder (beide duerfen nullptr sein):
// Jeder Vektor erhaelt eine Verlaesslichkeit aus der Bildstruktur (glatte
// Flaechen = unsicher) und, wenn beide Richtungen vorliegen, aus deren
// Uebereinstimmung. Unsichere Vektoren werden durch die Bewegung der
// verlaesslichen Umgebung ersetzt (normalisierte Faltung, mehrere Radien).
// I0 = Luminanz des aktuellen Bildes in Fluss-Aufloesung.
void RefineFlows(const Image& I0, Flow* fwd, Flow* bwd);

// Wendet Bewegungsunschaerfe an.
//   fwd: Fluss aktuelles -> naechstes Bild (darf nullptr sein)
//   bwd: Fluss aktuelles -> vorheriges Bild (darf nullptr sein)
//   flowScale: Faktor Fluss-Aufloesung -> Bildaufloesung (z. B. 2 bei halber Aufloesung)
void ApplyMotionBlur(const RGBAImage& src, const Flow* fwd, const Flow* bwd,
                     int flowScale, const BlurParams& p, RGBAImage& dst);

// Farbcodierte Darstellung des Vektorfelds (Farbton = Richtung, Helligkeit = Staerke).
void VisualizeFlow(const Flow& flow, int flowScale, int outW, int outH, RGBAImage& dst);

} // namespace fmb
