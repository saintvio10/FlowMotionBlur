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
#include <cmath>
#include <algorithm>

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
    // Wie resize, aber ohne Nullsetzen, wenn die Groesse schon passt (der Inhalt
    // wird anschliessend vollstaendig ueberschrieben). Spart Zeit bei grossen Bildern.
    void ensure(int width, int height) {
        w = width; h = height;
        const size_t n = static_cast<size_t>(w) * h * 4;
        if (d.size() != n) d.resize(n);
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
    const struct Flow* initFlow = nullptr; // optional: Startfeld (Groesse wie I0), ersetzt initU/V
    bool  occlusions     = true;   // Kamera-Ablauf: Verdeckungen korrigieren (ResolveOcclusions)
    bool  localHalf      = true;   // Kamera-Ablauf: feines Tracking + Auffuellen in halber Aufloesung
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
    float minMotion  = 0.f;  // Bewegung (px pro Bild, volle Aufloesung) bis zu der nichts verwischt;
                             // darueber steigt die Unschaerfe weich an
    // Nur Modus "Natuerlich":
    float phase      = 0.f;  // Lage des Verschlusses: 0 = mittig, +1 = vorlaufend (jetzt bis
                              // spaeter), -1 = nachlaufend (frueher bis jetzt)
    bool  soft       = false; // weicher Verschluss: Belichtung an den Enden weich ein-/ausblenden
};

// Bahn eines Pixels waehrend der Belichtung (Modus "Natuerlich"), gemeinsam fuer
// CPU, OpenCL und Metal: Geschwindigkeit V (Pixel pro Bild), Belichtung s in [s0, s1]
// (in Bildern, 0 = jetzt). Ein Pixel sammelt den Inhalt, der zum Zeitpunkt s bei ihm
// ankommt; der liegt jetzt bei x - V*s.
// f/b: Vorwaerts-/Rueckwaertsbewegung in Pixeln der vollen Aufloesung (hasF/hasB).
struct BlurPath { float vx, vy, s0, s1, extent, total; };
inline BlurPath MakeBlurPath(bool hasF, bool hasB, float fu, float fv, float bu, float bv, const BlurParams& p,
                             float maxRadius)
{
    BlurPath r{};
    if (hasF && hasB) { r.vx = 0.5f * (fu - bu); r.vy = 0.5f * (fv - bv); }
    else if (hasF) { r.vx = fu; r.vy = fv; }
    else             { r.vx = -bu; r.vy = -bv; }
    if (p.minMotion > 0.f) {
        const float l = std::sqrt(r.vx * r.vx + r.vy * r.vy);
        const float k = l > p.minMotion ? (l - p.minMotion) / l : 0.f;
        r.vx *= k; r.vy *= k;
    }
    const float half = 0.5f * p.shutter;
    r.s0 = half * (-1.f + p.phase); r.s1 = half * (1.f + p.phase);
    const float speed = std::sqrt(r.vx * r.vx + r.vy * r.vy);
    float l0 = speed * std::fabs(r.s0), l1 = speed * std::fabs(r.s1);
    float L = std::max(l0, l1);
    if (L > maxRadius) {
        const float k = maxRadius / L;
        r.vx *= k; r.vy *= k; l0 *= k; l1 *= k; L = maxRadius;
    }
    r.extent = L; r.total = l0 + l1;
    return r;
}
// Gewicht eines Samples bei t in [-1, 1] (weicher Verschluss: Rampen auf den aeusseren 35 %)
inline float ShutterWeight(float t, bool soft)
{
    if (!soft) return 1.f;
    const float w = (1.f - std::fabs(t)) / 0.35f;
    return w < 1.f ? (w > 0.f ? w : 0.f) : 1.f;
}

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

// Startfeld fuer die groebste Pyramidenstufe (w x h) aus FlowParams.
void CoarseInit(const FlowParams& p, int I0w, int w, int h, Flow& out);

// Bestimmt die Kamerabewegung von I0 nach I1 direkt aus den Bildern als
// Homographie (Zoom, Drehung, Perspektive), grob-nach-fein mit robuster
// Gewichtung. H arbeitet in normierten Koordinaten (siehe HomographyField).
// initDx/initDy: Startverschiebung in Pixeln von I0. residual: mittlerer
// Restfehler (Helligkeit 0..1). false, wenn die Schaetzung scheitert.
bool AlignHomography(const Image& I0, const Image& I1, float initDx, float initDy,
                     double H[9], double* residual = nullptr, const double* extraInit = nullptr);

// Fotometrischer Restfehler einer Homographie (ohne Optimierung), zum Vergleich
// verschiedener Kandidaten. Kleiner = besser.
double HomographyCost(const Image& I0, const Image& I1, const double H[9]);

// Umkehrung einer Homographie (normiert auf h33 = 1). false, wenn singulaer.
bool InvertHomography(const double H[9], double Hi[9]);

// Vektorfeld (w x h, Pixel) einer Homographie aus AlignHomography.
void HomographyField(const double H[9], int w, int h, Flow& out);

// Waehlt fuer jede Bildstelle die Bewegung, die am besten zu den Bildern passt:
// Kameramodell (mf/mb), feines Tracking (fwd/bwd, werden ueberschrieben) oder
// keine Bewegung. Inext/Iprev: Nachbarbilder (Luminanz, wie I0), je optional.
struct SelectStats { float model = 0, local = 0, zero = 0; };
void SelectCandidates(const Image& I0, const Image* Inext, const Image* Iprev,
                      Flow* fwd, Flow* bwd, const Flow* mf, const Flow* mb, SelectStats* stats = nullptr);

// Kompletter Ablauf fuer Kamerabewegung (Drohne, Schwenk, Fahrt):
//  1. Kamerabewegung direkt aus den Bildern (AlignHomography)
//  2. feines Tracking mit dem Modell als Startfeld (flowFn = CPU oder GPU)
//  3. unsichere Vektoren auffuellen (RefineFlows)
//  4. je Bildstelle Modell / Tracking / keine Bewegung waehlen (SelectCandidates)
// In/Ip: Luminanz des naechsten/vorherigen Bildes (wie I0), je optional.
struct CameraInfo {
    bool modelF = false, modelB = false;
    double resF = 0, resB = 0;
    // Diagnose: Vorsuche, Kosten von Modell und "keine Bewegung", Grund bei Ablehnung
    float shiftFx = 0, shiftFy = 0, shiftBx = 0, shiftBy = 0;
    double costF = 0, costB = 0, zeroF = 0, zeroB = 0;
    const char* whyF = "-"; const char* whyB = "-";
    bool priorUsedF = false, priorUsedB = false; // Startwert vom vorherigen Bild genutzt
    float occluded = 0;                          // Anteil korrigierter Verdeckungen
    double Hf[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }, Hb[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    SelectStats sel;
    // Zeitaufteilung in ms (Diagnose): Kameramodell, feines Tracking, Auswahl, Verdeckungen
    double msModel = 0, msTrack = 0, msSelect = 0, msOcclusion = 0;
    bool reusedB = false; // Rueckwaertsbewegung vom vorherigen Bild uebernommen
};
using FlowFn = std::function<void(const Image&, const Image&, const FlowParams&, Flow&)>;

// Kamerabewegung des vorherigen Bildes als Startwert: Die Kamera bewegt sich von
// Bild zu Bild kaum anders. Passt der Startwert, entfaellt die aufwendige Suche
// ueber viele Ausgangslagen (schneller, ruhiger).
struct CameraPrior {
    bool validF = false, validB = false;
    double Hf[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }, Hb[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
};

// bwdHint (optional): Rueckwaertsbewegung, die schon bekannt ist - die Umkehrung der
// Vorwaertsbewegung des vorherigen Bildes (InvertFlow). Dann entfaellt das feine
// Tracking rueckwaerts; Luecken (NaN) werden aus dem Kameramodell gefuellt.
void CameraFlows(const Image& I0, const Image* In, const Image* Ip, const FlowParams& p,
                 const FlowFn& flowFn, Flow* fwd, Flow* bwd, CameraInfo* info = nullptr,
                 const CameraPrior* prior = nullptr, const Flow* bwdHint = nullptr);

// Kehrt ein Vektorfeld um: f zeigt von Bild A nach B, out von B nach A (gleiche Groesse).
// Stellen in B, die in A nicht vorkommen (frei werdender Inhalt), sind NaN.
// Gibt den Anteil gueltiger Stellen zurueck.
float InvertFlow(const Flow& f, Flow& out);


// Verdeckungen: Wird eine Stelle im naechsten (oder vorherigen) Bild verdeckt bzw.
// erst freigelegt, ist eine der beiden Richtungen falsch. Wo vorwaerts und
// rueckwaerts nicht zusammenpassen, wird nur die Richtung verwendet, die besser zu
// den Bildern passt (die andere wird als Spiegelbild gesetzt).
// Rueckgabe: Anteil der korrigierten Stellen.
// mf/mb (optional): Kameramodell; dessen eigene Asymmetrie (Perspektive) gilt nicht als Verdeckung.
float ResolveOcclusions(const Image& I0, const Image& In, const Image& Ip, Flow& fwd, Flow& bwd,
                        const Flow* mf = nullptr, const Flow* mb = nullptr);

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
