// FlowMotionBlur.cpp - SDK-Anbindung (After Effects Effect API).
//
// Premiere Pro laedt Effekte, die mit dem After-Effects-SDK gebaut sind,
// direkt. Dieses Plugin funktioniert deshalb in Premiere Pro UND After Effects.
//
// Ablauf pro Frame:
//   1a. Automatisch: vorheriges/naechstes Bild holen, optischen Fluss
//       berechnen (optional gelenkt durch Track Points)
//   1b. Vektor-Ebene: Bewegungsvektoren aus einer anderen Ebene lesen
//   2.  Bild entlang der Vektoren verwischen
// Fluss und Unschaerfe laufen auf der GPU (OpenCL), sonst auf der CPU.

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "SPBasic.h"
#include "PrSDKAESupport.h"

#include "FlowMotionBlur.h"
#include "../core/FlowCore.h"
#include "../gpu/GpuBackend.h"
#include "DiagText.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <new>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <fstream>

// ---------------------------------------------------------------------------
// Suite-Helfer (ohne Abhaengigkeit von den Util-Dateien des SDK)
// ---------------------------------------------------------------------------

template <class SuiteT>
class Suite {
public:
    Suite(const PF_InData* in_data, const char* name, int32_t version)
        : basic_(in_data->pica_basicP), name_(name), version_(version)
    {
        const void* p = nullptr;
        if (basic_ && basic_->AcquireSuite(name_, version_, &p) == kSPNoError)
            suite_ = static_cast<const SuiteT*>(p);
    }
    ~Suite() { if (suite_) basic_->ReleaseSuite(name_, version_); }
    const SuiteT* operator->() const { return suite_; }
    explicit operator bool() const { return suite_ != nullptr; }
private:
    SPBasicSuite* basic_;
    const char*   name_;
    int32_t       version_;
    const SuiteT* suite_ = nullptr;
};

static bool IsPremiere(const PF_InData* in_data) { return in_data->appl_id == 'PrMr'; }

static inline double FixedToDouble(PF_Fixed v) { return static_cast<double>(v) / 65536.0; }

// Verkleinerungsfaktor der Vorschau (1 = volle Aufloesung, 2 = halbe, ...)
static double DownsampleX(const PF_InData* in_data)
{
    const PF_RationalScale& s = in_data->downsample_x;
    return (s.num > 0 && s.den > 0) ? static_cast<double>(s.den) / s.num : 1.0;
}
static double DownsampleY(const PF_InData* in_data)
{
    const PF_RationalScale& s = in_data->downsample_y;
    return (s.num > 0 && s.den > 0) ? static_cast<double>(s.den) / s.num : 1.0;
}

// ---------------------------------------------------------------------------
// Pixel-Formate
// ---------------------------------------------------------------------------

enum class PixKind { AE_ARGB8, AE_ARGB16, AE_ARGB32F, PR_BGRA8, PR_BGRA32F, Unknown };

static int BytesPerPixel(PixKind k)
{
    switch (k) {
    case PixKind::AE_ARGB8: case PixKind::PR_BGRA8: return 4;
    case PixKind::AE_ARGB16: return 8;
    case PixKind::AE_ARGB32F: case PixKind::PR_BGRA32F: return 16;
    default: return 0;
    }
}

// Ermittelt das Pixelformat. Die Angabe des Hosts wird gegen das tatsaechliche
// Speicherlayout (Bytes pro Zeile / Breite) geprueft. Passt sie nicht oder fehlt
// sie (in Premiere nicht immer verfuegbar), entscheidet das Speicherlayout.
static PixKind GetPixKind(PF_InData* in_data, PF_EffectWorld* world)
{
    if (!world || !world->data || world->width <= 0) return PixKind::Unknown;

    PF_PixelFormat fmt = PF_PixelFormat_INVALID;
    {
        Suite<PF_WorldSuite2> ws(in_data, kPFWorldSuite, kPFWorldSuiteVersion2);
        if (ws) ws->PF_GetPixelFormat(world, &fmt);
    }
    const int f = static_cast<int>(fmt);

    PixKind fromHost = PixKind::Unknown;
    if      (f == static_cast<int>(PrPixelFormat_BGRA_4444_8u))  fromHost = PixKind::PR_BGRA8;
    else if (f == static_cast<int>(PrPixelFormat_BGRA_4444_32f)) fromHost = PixKind::PR_BGRA32F;
    else if (f == static_cast<int>(PF_PixelFormat_ARGB128))      fromHost = PixKind::AE_ARGB32F;
    else if (f == static_cast<int>(PF_PixelFormat_ARGB64))       fromHost = PixKind::AE_ARGB16;
    else if (f == static_cast<int>(PF_PixelFormat_ARGB32))       fromHost = PixKind::AE_ARGB8;

    const long rb = world->rowbytes < 0 ? -static_cast<long>(world->rowbytes) : static_cast<long>(world->rowbytes);
    const long bpp = rb / world->width; // Bytes pro Pixel (+ evtl. Zeilen-Padding)

    const int hb = BytesPerPixel(fromHost);
    if (hb > 0 && bpp >= hb && bpp < 2 * hb) return fromHost;

    // Premiere: angemeldet ist nur BGRA 32 Bit Float; 8 Bit ist Premieres
    // Standardformat ARGB (gleiches Layout wie in After Effects).
    if (bpp >= 16) return IsPremiere(in_data) ? PixKind::PR_BGRA32F : PixKind::AE_ARGB32F;
    if (bpp >= 8)  return PixKind::AE_ARGB16;
    if (bpp >= 4)  return PixKind::AE_ARGB8;
    return PixKind::Unknown;
}

static inline unsigned char ToU8(float v)
{
    v = v * 255.f + 0.5f;
    return static_cast<unsigned char>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
}
static inline A_u_short ToU16(float v)
{
    v = v * PF_MAX_CHAN16 + 0.5f;
    return static_cast<A_u_short>(v < 0.f ? 0.f : (v > PF_MAX_CHAN16 ? PF_MAX_CHAN16 : v));
}

// Premiere arbeitet mit nicht vormultipliziertem Alpha, AE mit vormultipliziertem.
// Intern wird vormultipliziert gerechnet, damit transparente Kanten sauber bleiben.
// Fuer Vektor-Ebenen (premultiply = false) werden die Rohwerte gelesen.
static void ReadWorld(const PF_EffectWorld* w, PixKind kind, fmb::RGBAImage& out, bool straightAlpha, bool premultiply = true)
{
    out.resize(w->width, w->height);
    const bool straight = premultiply && straightAlpha;

    fmb::ParallelFor(0, w->height, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const char* row = static_cast<const char*>(static_cast<const void*>(w->data)) +
                              static_cast<ptrdiff_t>(y) * w->rowbytes;
            for (int x = 0; x < w->width; ++x) {
                float* o = out.px(x, y);
                switch (kind) {
                case PixKind::AE_ARGB8: {
                    const PF_Pixel* p = reinterpret_cast<const PF_Pixel*>(row) + x;
                    o[0] = p->red / 255.f; o[1] = p->green / 255.f; o[2] = p->blue / 255.f; o[3] = p->alpha / 255.f;
                    break; }
                case PixKind::AE_ARGB16: {
                    const PF_Pixel16* p = reinterpret_cast<const PF_Pixel16*>(row) + x;
                    const float k = 1.f / PF_MAX_CHAN16;
                    o[0] = p->red * k; o[1] = p->green * k; o[2] = p->blue * k; o[3] = p->alpha * k;
                    break; }
                case PixKind::AE_ARGB32F: {
                    const PF_PixelFloat* p = reinterpret_cast<const PF_PixelFloat*>(row) + x;
                    o[0] = p->red; o[1] = p->green; o[2] = p->blue; o[3] = p->alpha;
                    break; }
                case PixKind::PR_BGRA8: {
                    const unsigned char* p = reinterpret_cast<const unsigned char*>(row) + x * 4;
                    o[0] = p[2] / 255.f; o[1] = p[1] / 255.f; o[2] = p[0] / 255.f; o[3] = p[3] / 255.f;
                    break; }
                case PixKind::PR_BGRA32F: {
                    const float* p = reinterpret_cast<const float*>(row) + x * 4;
                    o[0] = p[2]; o[1] = p[1]; o[2] = p[0]; o[3] = p[3];
                    break; }
                default:
                    o[0] = o[1] = o[2] = o[3] = 0.f;
                }
                if (straight) { o[0] *= o[3]; o[1] *= o[3]; o[2] *= o[3]; }
            }
        }
    });
}

static void WriteWorld(const fmb::RGBAImage& img, PixKind kind, PF_EffectWorld* w, bool straightAlpha)
{
    const int width = std::min<int>(img.w, w->width);
    const int height = std::min<int>(img.h, w->height);
    const bool straight = straightAlpha;

    fmb::ParallelFor(0, height, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            char* row = static_cast<char*>(static_cast<void*>(w->data)) + static_cast<ptrdiff_t>(y) * w->rowbytes;
            for (int x = 0; x < width; ++x) {
                const float* s = img.px(x, y);
                float r = s[0], g = s[1], b = s[2], a = s[3];
                if (straight && a > 1e-6f) { r /= a; g /= a; b /= a; }
                switch (kind) {
                case PixKind::AE_ARGB8: {
                    PF_Pixel* p = reinterpret_cast<PF_Pixel*>(row) + x;
                    p->red = ToU8(r); p->green = ToU8(g); p->blue = ToU8(b); p->alpha = ToU8(a);
                    break; }
                case PixKind::AE_ARGB16: {
                    PF_Pixel16* p = reinterpret_cast<PF_Pixel16*>(row) + x;
                    p->red = ToU16(r); p->green = ToU16(g); p->blue = ToU16(b); p->alpha = ToU16(a);
                    break; }
                case PixKind::AE_ARGB32F: {
                    PF_PixelFloat* p = reinterpret_cast<PF_PixelFloat*>(row) + x;
                    p->red = r; p->green = g; p->blue = b; p->alpha = a;
                    break; }
                case PixKind::PR_BGRA8: {
                    unsigned char* p = reinterpret_cast<unsigned char*>(row) + x * 4;
                    p[0] = ToU8(b); p[1] = ToU8(g); p[2] = ToU8(r); p[3] = ToU8(a);
                    break; }
                case PixKind::PR_BGRA32F: {
                    float* p = reinterpret_cast<float*>(row) + x * 4;
                    p[0] = b; p[1] = g; p[2] = r; p[3] = a;
                    break; }
                default: break;
                }
            }
        }
    });
}

static double AlphaSum(const fmb::RGBAImage& img)
{
    double s = 0;
    for (size_t i = 3; i < img.d.size(); i += 16) s += img.d[i]; // jedes 4. Pixel reicht
    return s;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

static PF_Err About(PF_InData* in_data, PF_OutData* out_data)
{
    (void)in_data;
    const std::string dev = fmb::gpu::DeviceName();
    std::snprintf(out_data->return_msg, sizeof(out_data->return_msg),
                  "%s v%d.%d\r%s\rGPU: %s", FMB_NAME, FMB_MAJOR_VERSION, FMB_MINOR_VERSION,
                  FMB_DESCRIPTION, dev.empty() ? "keine (CPU)" : dev.c_str());
    return PF_Err_NONE;
}

static PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data)
{
    out_data->my_version = PF_VERSION(FMB_MAJOR_VERSION, FMB_MINOR_VERSION, FMB_BUG_VERSION,
                                      FMB_STAGE_VERSION, FMB_BUILD_VERSION);

    // Muss zu AE_Effect_Global_OutFlags in FlowMotionBlurPiPL.r passen (0x02000006).
    out_data->out_flags = PF_OutFlag_WIDE_TIME_INPUT |   // liest Nachbarframes
                          PF_OutFlag_NON_PARAM_VARY |    // Ergebnis haengt vom Bildinhalt ab
                          PF_OutFlag_DEEP_COLOR_AWARE;   // 16 Bit in AE
    out_data->out_flags2 = 0;

    if (IsPremiere(in_data)) {
        // Premiere: zusaetzlich BGRA 32 Bit Float anbieten.
        Suite<PF_PixelFormatSuite1> pfs(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1);
        if (pfs) {
            pfs->ClearSupportedPixelFormats(in_data->effect_ref);
            // Nur 32 Bit Float anmelden. 8 Bit liefert Premiere dann im Standardformat
            // ARGB; so sind beide Faelle allein am Speicherlayout unterscheidbar.
            pfs->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
        }
    }
    return PF_Err_NONE;
}

static PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Blur Amount (Shutter)", 0, 4, 0, 2, 0.5, PF_Precision_HUNDREDTHS,
                         0, 0, AMOUNT_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER("Max Samples", 3, 256, 3, 64, 32, MAX_SAMPLES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Bewegungsart", 2, MOTION_CAMERA,
                 "Kamera (Drohne, Schwenk, Fahrt)|Frei (bewegte Objekte)", MOTION_TYPE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Unschaerfe-Modus", 2, MODE_NATURAL,
                 "Natuerlich (Kamera)|Objekte (Vordergrund verschmiert)", BLUR_MODE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Analyse-Qualitaet", 3, QUALITY_NORMAL,
                 "Schnell (1/4)|Normal (1/2)|Hoch (voll)", QUALITY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER("Vektor-Glaettung", 0, 10, 0, 10, 2, SMOOTHING_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("RGB+A-Tracking", FALSE, 0, RGBA_TRACKING_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("GPU verwenden", TRUE, 0, USE_GPU_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Ansicht", 3, VIEW_RESULT, "Ergebnis|Bewegungsvektoren|Diagnose", VIEW_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Vektor-Quelle", 2, SOURCE_TRACKING, "Automatisch (Tracking)|Vektor-Ebene",
                 VECTOR_SOURCE_DISK_ID);

    // --- Eigene Bewegungsvektoren -------------------------------------------
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Eigene Bewegungsvektoren", VEC_TOPIC_START_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER("Vektor-Ebene", PF_LayerDefault_NONE, VEC_LAYER_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Max. Verschiebung (px)", 1, 4096, 1, 256, 32, PF_Precision_TENTHS,
                         0, 0, VEC_MAX_DISP_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Kodierung", 2, ENCODING_NORMALIZED,
                 "Normalisiert (0.5 = keine Bewegung)|Pixel (32-Bit Float)", VEC_ENCODING_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Y umkehren", FALSE, 0, VEC_FLIP_Y_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(VEC_TOPIC_END_DISK_ID);

    // --- Track Points -------------------------------------------------------
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Track Points", TP_TOPIC_START_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Einflussradius (px)", 5, 2000, 5, 400, 80, PF_Precision_INTEGER,
                         0, 0, TP_RADIUS_DISK_ID);

    for (int i = 0; i < FMB_NUM_TRACK_POINTS; ++i) {
        char onName[32], posName[32];
        std::snprintf(onName, sizeof(onName), "Punkt %d aktiv", i + 1);
        std::snprintf(posName, sizeof(posName), "Punkt %d", i + 1);

        AEFX_CLR_STRUCT(def);
        PF_ADD_CHECKBOXX(onName, FALSE, 0, TP_FIRST_DISK_ID + 2 * i);

        AEFX_CLR_STRUCT(def);
        PF_ADD_POINT(posName, 50, 50, FALSE, TP_FIRST_DISK_ID + 2 * i + 1);
    }

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(TP_TOPIC_END_DISK_ID);

    out_data->num_params = FMB_NUM_PARAMS;
    return err;
}

// Liest einen Punkt-Parameter zu einem anderen Zeitpunkt (in Pixeln des
// aktuellen, evtl. verkleinerten Bildes).
static bool PointAt(PF_InData* in_data, int index, A_long time, double& x, double& y)
{
    PF_ParamDef ck;
    AEFX_CLR_STRUCT(ck);
    if (PF_CHECKOUT_PARAM(in_data, index, time, in_data->time_step, in_data->time_scale, &ck) != PF_Err_NONE)
        return false;
    x = FixedToDouble(ck.u.td.x_value) / DownsampleX(in_data);
    y = FixedToDouble(ck.u.td.y_value) / DownsampleY(in_data);
    PF_CHECKIN_PARAM(in_data, &ck);
    return true;
}

// Track Points fuer den Fluss zum Frame bei current_time + offset.
static std::vector<fmb::Guide> CollectGuides(PF_InData* in_data, PF_ParamDef* params[], A_long offset, int factor)
{
    std::vector<fmb::Guide> out;
    for (int i = 0; i < FMB_NUM_TRACK_POINTS; ++i) {
        if (!params[FMB_TP_ON(i)]->u.bd.value) continue;
        double x0, y0, x1, y1;
        if (!PointAt(in_data, FMB_TP_POS(i), in_data->current_time, x0, y0)) continue;
        if (!PointAt(in_data, FMB_TP_POS(i), in_data->current_time + offset, x1, y1)) continue;
        fmb::Guide g;
        g.x = static_cast<float>(x0 / factor);
        g.y = static_cast<float>(y0 / factor);
        g.dx = static_cast<float>((x1 - x0) / factor);
        g.dy = static_cast<float>((y1 - y0) / factor);
        out.push_back(g);
    }
    return out;
}

// Liest die Vektor-Ebene und wandelt sie in ein Vektorfeld (Pixel pro Frame,
// Aufloesung des aktuellen Bildes). false, wenn keine Ebene gewaehlt ist.
static bool ReadVectorLayer(PF_InData* in_data, PF_ParamDef* params[], int outW, int outH,
                            fmb::Flow& flow, PF_Err& err)
{
    PF_ParamDef ck;
    AEFX_CLR_STRUCT(ck);
    err = PF_CHECKOUT_PARAM(in_data, FMB_VEC_LAYER, in_data->current_time,
                            in_data->time_step, in_data->time_scale, &ck);
    if (err != PF_Err_NONE) {
        if (err != PF_Interrupt_CANCEL) err = PF_Err_NONE; // keine Ebene -> Tracking
        return false;
    }

    bool ok = false;
    PF_EffectWorld* w = &ck.u.ld;
    if (w->data && w->width > 0 && w->height > 0) {
        PixKind kind = GetPixKind(in_data, w);
        if (kind != PixKind::Unknown) {
            fmb::RGBAImage vec;
            ReadWorld(w, kind, vec, IsPremiere(in_data), false);

            const bool  normalized = params[FMB_VEC_ENCODING]->u.pd.value == ENCODING_NORMALIZED;
            const float maxDisp = static_cast<float>(params[FMB_VEC_MAX_DISP]->u.fs_d.value);
            const float flipY = params[FMB_VEC_FLIP_Y]->u.bd.value ? -1.f : 1.f;
            // Vektoren sind in Pixeln der vollen Aufloesung angegeben; bei
            // reduzierter Vorschau-Aufloesung werden sie entsprechend kleiner.
            const float kx = static_cast<float>(1.0 / DownsampleX(in_data));
            const float ky = static_cast<float>(1.0 / DownsampleY(in_data)) * flipY;
            const float sx = static_cast<float>(vec.w) / outW, sy = static_cast<float>(vec.h) / outH;

            flow.u.resize(outW, outH);
            flow.v.resize(outW, outH);
            fmb::ParallelFor(0, outH, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < outW; ++x) {
                        int vx = std::min(vec.w - 1, static_cast<int>(x * sx));
                        int vy = std::min(vec.h - 1, static_cast<int>(y * sy));
                        const float* p = vec.px(vx, vy);
                        float u = normalized ? (p[0] - 0.5f) * 2.f * maxDisp : p[0];
                        float v = normalized ? (p[1] - 0.5f) * 2.f * maxDisp : p[1];
                        flow.u.row(y)[x] = u * kx;
                        flow.v.row(y)[x] = v * ky;
                    }
            });
            ok = true;
        }
    }
    PF_Err err2 = PF_CHECKIN_PARAM(in_data, &ck);
    if (err == PF_Err_NONE) err = err2;
    return ok;
}

struct Engine {
    bool useGpu;
    mutable bool gpuUsed = false;
    void Flow(const fmb::Image& a, const fmb::Image& b, const fmb::FlowParams& fp,
              const std::vector<fmb::Guide>* g, fmb::Flow& out) const
    {
        if (useGpu && fmb::gpu::ComputeFlow(a, b, fp, out, g)) { gpuUsed = true; return; }
        fmb::ComputeFlow(a, b, fp, out, g);
    }
    void Blur(const fmb::RGBAImage& src, const fmb::Flow* f, const fmb::Flow* b, int scale,
              const fmb::BlurParams& bp, fmb::RGBAImage& dst) const
    {
        if (useGpu && fmb::gpu::ApplyMotionBlur(src, f, b, scale, bp, dst)) { gpuUsed = true; return; }
        fmb::ApplyMotionBlur(src, f, b, scale, bp, dst);
    }
};

// Diagnose-Informationen zu einem Nachbarframe
struct NbInfo {
    std::string status = "NICHT ABGEFRAGT";
    double diff = 0;    // mittlerer Helligkeitsunterschied zum aktuellen Bild
    double motion = 0;  // mittlere erkannte Bewegung in Pixeln
    double shift = 0;   // Ergebnis der groben Vorsuche in Pixeln
};

static std::string Num(double v, int prec = 1)
{
    char b[64]; std::snprintf(b, sizeof(b), "%.*f", prec, v); return b;
}

static const char* KindName(PixKind k)
{
    switch (k) {
    case PixKind::AE_ARGB8: return "ARGB 8";  case PixKind::AE_ARGB16: return "ARGB 16";
    case PixKind::AE_ARGB32F: return "ARGB 32F"; case PixKind::PR_BGRA8: return "BGRA 8";
    case PixKind::PR_BGRA32F: return "BGRA 32F"; default: return "UNBEKANNT";
    }
}

// ---------------------------------------------------------------------------
// Diagnose-Export: Bilder und Werte so speichern, wie das Plugin sie erhaelt
// ---------------------------------------------------------------------------
static std::filesystem::path DiagDir()
{
#if defined(_WIN32)
    const char* base = std::getenv("APPDATA");
    return std::filesystem::path(base ? base : ".") / "FlowMotionBlur" / "Diagnose";
#else
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home ? home : ".") / "Library" / "Application Support" / "FlowMotionBlur" / "Diagnose";
#endif
}

static bool WritePGM(const std::filesystem::path& p, const fmb::Image& im)
{
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f << "P5\n" << im.w << " " << im.h << "\n255\n";
    std::vector<unsigned char> row(im.w);
    for (int y = 0; y < im.h; ++y) {
        for (int x = 0; x < im.w; ++x) {
            float v = im.row(y)[x] * 255.f + 0.5f;
            row[x] = static_cast<unsigned char>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        f.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
    return static_cast<bool>(f);
}

// Bilineares Skalieren eines RGBA-Bildes (fuer Nachbarframes in anderer Groesse).
static void ResizeRGBA(const fmb::RGBAImage& src, int w, int h, fmb::RGBAImage& dst)
{
    dst.resize(w, h);
    const float sx = static_cast<float>(src.w) / w, sy = static_cast<float>(src.h) / h;
    fmb::ParallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                float fx = std::min(std::max((x + 0.5f) * sx - 0.5f, 0.f), static_cast<float>(src.w - 1));
                float fy = std::min(std::max((y + 0.5f) * sy - 0.5f, 0.f), static_cast<float>(src.h - 1));
                int x0 = static_cast<int>(fx), y0i = static_cast<int>(fy);
                int x1 = std::min(x0 + 1, src.w - 1), y1i = std::min(y0i + 1, src.h - 1);
                float ax = fx - x0, ay = fy - y0i;
                const float *a = src.px(x0, y0i), *b = src.px(x1, y0i), *c = src.px(x0, y1i), *d = src.px(x1, y1i);
                float* o = dst.px(x, y);
                for (int k = 0; k < 4; ++k)
                    o[k] = (a[k] * (1 - ax) + b[k] * ax) * (1 - ay) + (c[k] * (1 - ax) + d[k] * ax) * ay;
            }
    });
}

// Holt den Frame bei current_time + offset und berechnet den Fluss dorthin.
// Gibt false zurueck, wenn kein brauchbarer Nachbarframe existiert (Clip-Anfang/-Ende).
// computeFlow = false: nur die Luminanz des Nachbarframes liefern (nbLuma),
// der Fluss wird dann gemeinsam fuer beide Richtungen berechnet (Kamera-Modus).
static bool NeighbourFlow(PF_InData* in_data, A_long offset, const fmb::RGBAImage& cur,
                          const fmb::Image& curLuma, int factor, const fmb::FlowParams& fp,
                          float alphaWeight, double curAlpha, const std::vector<fmb::Guide>& guides,
                          const Engine& eng, fmb::Flow& flow, PF_Err& err, NbInfo& info,
                          bool computeFlow, fmb::Image& nbLuma)
{
    PF_ParamDef ck;
    AEFX_CLR_STRUCT(ck);
    err = PF_CHECKOUT_PARAM(in_data, FMB_INPUT, in_data->current_time + offset,
                            in_data->time_step, in_data->time_scale, &ck);
    if (err != PF_Err_NONE) {
        info.status = "ABRUF FEHLGESCHLAGEN (CODE " + std::to_string(static_cast<int>(err)) + ")";
        // Abbruch durch den Benutzer weitergeben, alles andere = kein Nachbarframe
        if (err != PF_Interrupt_CANCEL) err = PF_Err_NONE;
        return false;
    }

    bool ok = false;
    PF_EffectWorld* w = &ck.u.ld;
    if (!w->data || w->width <= 0 || w->height <= 0) {
        info.status = "KEIN BILD GELIEFERT";
    } else {
        PixKind kind = GetPixKind(in_data, w);
        if (kind == PixKind::Unknown) {
            info.status = "FORMAT UNBEKANNT";
        } else {
            fmb::RGBAImage nb;
            ReadWorld(w, kind, nb, IsPremiere(in_data));
            std::string sizeNote;
            if (nb.w != cur.w || nb.h != cur.h) {
                // Premiere liefert Nachbarframes evtl. in Clip- statt Sequenzgroesse
                sizeNote = " SKALIERT VON " + std::to_string(nb.w) + "X" + std::to_string(nb.h);
                fmb::RGBAImage tmp;
                ResizeRGBA(nb, cur.w, cur.h, tmp);
                nb = std::move(tmp);
            }
            // Ausserhalb des Clips liefern Hosts oft ein leeres (transparentes) Bild.
            if (curAlpha > 0.0 && AlphaSum(nb) <= 0.02 * curAlpha) {
                info.status = "TRANSPARENT (AUSSERHALB DES CLIPS)" + sizeNote;
            } else {
                fmb::LumaFromRGBA(nb, factor, nbLuma, alphaWeight);
                double d = 0;
                for (size_t i = 0; i < nbLuma.d.size(); ++i) d += std::fabs(nbLuma.d[i] - curLuma.d[i]);
                info.diff = nbLuma.d.empty() ? 0 : d / nbLuma.d.size();
                if (computeFlow) {
                    // Grobe Vorsuche: findet auch sehr grosse Bewegungen (Speed Ramps)
                    fmb::FlowParams fp2 = fp;
                    fmb::EstimateGlobalShift(curLuma, nbLuma, fp2.initU, fp2.initV);
                    info.shift = std::hypot(fp2.initU, fp2.initV) * factor;
                    eng.Flow(curLuma, nbLuma, fp2, guides.empty() ? nullptr : &guides, flow);
                }
                info.status = (info.diff < 1e-4 ? "OK, ABER GLEICHES BILD" : "OK") + sizeNote;
                ok = true;
            }
        }
    }
    PF_CHECKIN_PARAM(in_data, &ck);
    return ok;
}

static PF_Err Render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    PF_EffectWorld* input = &params[FMB_INPUT]->u.ld;

    const PixKind inKind = GetPixKind(in_data, input);
    const PixKind outKind = GetPixKind(in_data, output);
    if (inKind == PixKind::Unknown || outKind == PixKind::Unknown)
        return PF_COPY(input, output, NULL, NULL); // nie mit Fehler abbrechen

    const float amount    = static_cast<float>(params[FMB_AMOUNT]->u.fs_d.value);
    const int   samples   = params[FMB_MAX_SAMPLES]->u.sd.value;
    const int   quality   = params[FMB_QUALITY]->u.pd.value;
    const int   smoothing = params[FMB_SMOOTHING]->u.sd.value;
    const int   view      = params[FMB_VIEW]->u.pd.value;
    const bool  rgbaTrack = params[FMB_RGBA_TRACKING]->u.bd.value != 0;
    const int   source    = params[FMB_VECTOR_SOURCE]->u.pd.value;
    const float alphaWeight = rgbaTrack ? 0.5f : 0.f;
    const Engine eng{ params[FMB_USE_GPU]->u.bd.value != 0 };

    if (amount <= 0.f && view == VIEW_RESULT)
        return PF_COPY(input, output, NULL, NULL);

    try {
        fmb::RGBAImage cur;
        ReadWorld(input, inKind, cur, IsPremiere(in_data));

        fmb::BlurParams bp;
        bp.shutter = amount;
        bp.maxSamples = samples;
        bp.mode = (params[FMB_BLUR_MODE]->u.pd.value == MODE_OBJECTS) ? fmb::kBlurObjects : fmb::kBlurNatural;
        fmb::RGBAImage result;

        // --- Variante 1: Vektoren aus einer Ebene -------------------------
        if (source == SOURCE_LAYER) {
            fmb::Flow vec;
            if (ReadVectorLayer(in_data, params, cur.w, cur.h, vec, err)) {
                if (view == VIEW_VECTORS) fmb::VisualizeFlow(vec, 1, cur.w, cur.h, result);
                else                      eng.Blur(cur, &vec, nullptr, 1, bp, result);
                WriteWorld(result, outKind, output, IsPremiere(in_data));
                return err;
            }
            if (err) return err;
            // keine Ebene gewaehlt -> automatisches Tracking
        }

        // --- Variante 2: automatisches Tracking ----------------------------
        int factor = (quality == QUALITY_DRAFT) ? 4 : (quality == QUALITY_HIGH ? 1 : 2);
        // Bei reduzierter Vorschau-Aufloesung ist das Bild schon kleiner.
        const int ds = std::max(1, static_cast<int>(DownsampleX(in_data) + 0.5));
        factor = std::max(1, factor / ds);

        fmb::FlowParams fp;
        fp.smoothRadius = smoothing;
        fp.windowRadius = (factor >= 4) ? 2 : 3;
        fp.guideRadius  = static_cast<float>(params[FMB_TP_RADIUS]->u.fs_d.value / DownsampleX(in_data) / factor);

        fmb::Image curLuma;
        fmb::LumaFromRGBA(cur, factor, curLuma, alphaWeight);
        const double curAlpha = AlphaSum(cur);

        // Abstand zum Nachbarframe. Fehlt die Angabe, ca. 1/25 s verwenden
        // (frueher 1 Zeiteinheit -> praktisch derselbe Frame, keine Bewegung).
        const char* stepSrc = "TIME_STEP";
        A_long step = in_data->time_step;
        if (step == 0) { step = in_data->local_time_step; stepSrc = "LOCAL_TIME_STEP"; }
        if (step == 0) { step = std::max<A_long>(1, static_cast<A_long>(in_data->time_scale / 25)); stepSrc = "ERSATZ 1/25 S"; }
        if (step < 0) step = -step;

        const std::vector<fmb::Guide> gF = CollectGuides(in_data, params, +step, factor);
        const std::vector<fmb::Guide> gB = CollectGuides(in_data, params, -step, factor);

        // Im Kamera-Modus werden Track Points erst nach dem Modell angewendet,
        // sonst verzerren sie die Schaetzung der Kamerabewegung.
        const bool cameraMode = params[FMB_MOTION_TYPE]->u.pd.value != MOTION_FREE;
        const std::vector<fmb::Guide> noGuides;
        const std::vector<fmb::Guide>& gFtrack = cameraMode ? noGuides : gF;
        const std::vector<fmb::Guide>& gBtrack = cameraMode ? noGuides : gB;

        fmb::Flow fwd, bwd;
        fmb::Image lumF, lumB;
        NbInfo infoF, infoB;
        // Doppelte Bilder (Material mit weniger fps als die Sequenz, langsame
        // Abschnitte einer Speed Ramp): Ist der Nachbar identisch mit dem
        // aktuellen Bild, das naechste wirklich andere Bild holen (bis 4 weiter)
        // und die Bewegung spaeter durch den Abstand teilen.
        const double kDup = 0.0025; // mittlerer Helligkeitsunterschied unter 0.25 % = gleiches Bild
        int kF = 1, kB = 1;
        bool haveF = false, haveB = false;
        for (int k = 1; k <= 4; ++k) {
            NbInfo tmp;
            haveF = NeighbourFlow(in_data, +step * k, cur, curLuma, factor, fp, alphaWeight, curAlpha, gFtrack, eng, fwd, err, tmp, !cameraMode, lumF);
            if (err) return err;
            if ((err = PF_ABORT(in_data)) != PF_Err_NONE) return err;
            infoF = tmp; kF = k;
            if (!haveF || tmp.diff >= kDup) break;
        }
        for (int k = 1; k <= 4; ++k) {
            NbInfo tmp;
            haveB = NeighbourFlow(in_data, -step * k, cur, curLuma, factor, fp, alphaWeight, curAlpha, gBtrack, eng, bwd, err, tmp, !cameraMode, lumB);
            if (err) return err;
            if ((err = PF_ABORT(in_data)) != PF_Err_NONE) return err;
            infoB = tmp; kB = k;
            if (!haveB || tmp.diff >= kDup) break;
        }
        if (haveF && infoF.diff < kDup) kF = 1; // nirgends Bewegung: Abstand egal
        if (haveB && infoB.diff < kDup) kB = 1;
        // Wiederholt das Material jedes Bild r-mal (z. B. 30 fps in 60-fps-Sequenz:
        // AABBCC), enthaelt auch der direkt benachbarte andere Frame die Bewegung
        // von r Bildern. Daher beide Richtungen durch denselben Faktor teilen.
        const int rep = std::max(kF, kB);

        // Plausibilitaet der Nachbarbilder: Bei Zeitverzerrung (Speed Ramp,
        // Zeit-Neuzuordnung) liefert Premiere Nachbarbilder aus der unverzerrten
        // Zeit, also von einer ganz anderen Stelle des Clips. Erkennbar daran, dass
        // vorheriges und naechstes Bild sich aehneln, beide aber stark vom
        // aktuellen Bild abweichen. Dann nicht verwischen, sondern Hinweis zeigen.
        bool neighboursWrong = false;
        double diffNP = 0;
        if (haveF && haveB && lumF.w == lumB.w && lumF.h == lumB.h && !lumF.d.empty()) {
            for (size_t i = 0; i < lumF.d.size(); ++i) diffNP += std::fabs(lumF.d[i] - lumB.d[i]);
            diffNP /= lumF.d.size();
            const double dCur = std::min(infoF.diff, infoB.diff);
            neighboursWrong = dCur > 0.06 && diffNP < 0.4 * dCur;
        }
        if (neighboursWrong) {
            infoF.status = "FALSCH (ANDERE STELLE IM CLIP)";
            infoB.status = "FALSCH (ANDERE STELLE IM CLIP)";
            haveF = haveB = false;
        }

        // Bewegung auf einen Bildabstand umrechnen (doppelte Bilder)
        auto scaleFlow = [](fmb::Flow& f, int k) {
            if (k <= 1) return;
            const float inv = 1.f / k;
            for (float& v : f.u.d) v *= inv;
            for (float& v : f.v.d) v *= inv;
        };

        std::string modelState = "AUS (FREI)";
        fmb::CameraInfo camInfo;
        if (cameraMode && (haveF || haveB)) {
            // Kamerabewegung direkt aus den Bildern, feines Tracking ab diesem
            // Modell, Auswahl je Bildstelle (Modell / Tracking / keine Bewegung).
            fmb::CameraInfo ci;
            fmb::CameraFlows(curLuma, haveF ? &lumF : nullptr, haveB ? &lumB : nullptr, fp,
                             [&](const fmb::Image& a, const fmb::Image& b, const fmb::FlowParams& q, fmb::Flow& o) {
                                 eng.Flow(a, b, q, nullptr, o);
                             },
                             haveF ? &fwd : nullptr, haveB ? &bwd : nullptr, &ci);
            camInfo = ci;
            if (haveF) scaleFlow(fwd, rep);
            if (haveB) scaleFlow(bwd, rep);
            const bool okAll = (!haveF || ci.modelF) && (!haveB || ci.modelB);
            modelState = std::string(okAll ? "OK" : "TEILWEISE") + "  MODELL " + Num(ci.sel.model * 100, 0) +
                         "% TRACKING " + Num(ci.sel.local * 100, 0) + "% RUHE " + Num(ci.sel.zero * 100, 0) + "%";
            // Track Points zuletzt anwenden
            if (haveF && !gF.empty()) fmb::ApplyGuides(fwd, gF, fp.guideRadius, 1.f);
            if (haveB && !gB.empty()) fmb::ApplyGuides(bwd, gB, fp.guideRadius, 1.f);
        } else if (!cameraMode) {
            if (haveF) scaleFlow(fwd, rep);
            if (haveB) scaleFlow(bwd, rep);
            fmb::RefineFlows(curLuma, haveF ? &fwd : nullptr, haveB ? &bwd : nullptr);
        }
        // mittlere Bewegung fuer die Diagnose
        auto meanMotion = [&](const fmb::Flow& f) {
            double m = 0; for (size_t i = 0; i < f.u.d.size(); ++i) m += std::hypot(f.u.d[i], f.v.d[i]);
            return f.u.d.empty() ? 0.0 : m / f.u.d.size() * factor;
        };
        if (haveF) infoF.motion = meanMotion(fwd);
        if (haveB) infoB.motion = meanMotion(bwd);

        if (view == VIEW_VECTORS) {
            const fmb::Flow& show = haveF ? fwd : bwd;
            fmb::VisualizeFlow(show, factor, cur.w, cur.h, result);
        } else {
            eng.Blur(cur, haveF ? &fwd : nullptr, haveB ? &bwd : nullptr, factor, bp, result);
        }
        if (neighboursWrong && view != VIEW_DIAG) {
            std::vector<std::string> W;
            W.push_back("FLOW MOTION BLUR: FALSCHE NACHBARBILDER");
            W.push_back("SPEED RAMP / ZEITVERZERRUNG AM CLIP:");
            W.push_back("CLIP VERSCHACHTELN, EFFEKT AUF DIE");
            W.push_back("VERSCHACHTELTE SEQUENZ ANWENDEN");
            diag::DrawLines(result, W);
        }
        if (view == VIEW_DIAG) {
            std::vector<std::string> L;
            L.push_back("FLOW MOTION BLUR " + std::to_string(FMB_MAJOR_VERSION) + "." + std::to_string(FMB_MINOR_VERSION) + " - DIAGNOSE");
            L.push_back(std::string("HOST: ") + (IsPremiere(in_data) ? "PREMIERE" : "AFTER EFFECTS"));
            L.push_back(std::string("FORMAT EIN/AUS: ") + KindName(inKind) + " / " + KindName(outKind));
            L.push_back("BILD: " + std::to_string(cur.w) + "X" + std::to_string(cur.h) +
                        "  AUSGABE: " + std::to_string(output->width) + "X" + std::to_string(output->height));
            L.push_back("ZEIT: " + std::to_string(in_data->current_time) + "  SKALA: " + std::to_string(in_data->time_scale));
            L.push_back("TIME_STEP: " + std::to_string(in_data->time_step) + "  LOCAL: " + std::to_string(in_data->local_time_step));
            L.push_back("VERWENDET: " + std::to_string(step) + " (" + stepSrc + ")");
            L.push_back("VORSCHAU 1/" + Num(DownsampleX(in_data), 0) + "  ANALYSE 1/" + std::to_string(factor));
            L.push_back("KAMERA: " + modelState);
            L.push_back("VORWAERTS: " + infoF.status + (kF > 1 ? "  (" + std::to_string(kF) + " BILDER WEITER, DOPPELTE BILDER)" : ""));
            L.push_back("  UNTERSCHIED " + Num(infoF.diff * 100, 2) + "  VORSUCHE " + Num(infoF.shift) + "  BEWEGUNG " + Num(infoF.motion) + " PX");
            L.push_back("RUECKWAERTS: " + infoB.status + (kB > 1 ? "  (" + std::to_string(kB) + " BILDER WEITER, DOPPELTE BILDER)" : ""));
            L.push_back("  UNTERSCHIED " + Num(infoB.diff * 100, 2) + "  VORSUCHE " + Num(infoB.shift) + "  BEWEGUNG " + Num(infoB.motion) + " PX");
            L.push_back(std::string("GPU: ") + (eng.gpuUsed ? "JA" : (params[FMB_USE_GPU]->u.bd.value ? "NICHT VERFUEGBAR" : "AUS")));
            L.push_back("NACHBARN UNTEREINANDER: " + Num(diffNP * 100, 2) + (neighboursWrong ? "  -> ZEITVERZERRUNG, CLIP VERSCHACHTELN" : ""));
            if (cameraMode) {
                auto modelLines = [&](const char* name, bool have, float sx, float sy, const char* why,
                                      double cost, double zero, const double* Hm) {
                    if (!have) { L.push_back(std::string(name) + ": KEIN NACHBARBILD"); return; }
                    L.push_back(std::string(name) + ": VORSUCHE " + Num(sx * factor, 0) + "," + Num(sy * factor, 0) +
                                "  MODELL " + why);
                    L.push_back("  KOSTEN " + Num(cost * 1000, 2) + "  RUHE " + Num(zero * 1000, 2) +
                                "  SKAL " + Num(Hm[0] * Hm[4] - Hm[1] * Hm[3], 3) +
                                "  T " + Num(Hm[2], 3) + "," + Num(Hm[5], 3) + "  P " + Num(Hm[6], 3) + "," + Num(Hm[7], 3));
                };
                modelLines("MODELL VOR", haveF, camInfo.shiftFx, camInfo.shiftFy, camInfo.whyF, camInfo.costF, camInfo.zeroF, camInfo.Hf);
                modelLines("MODELL RUECK", haveB, camInfo.shiftBx, camInfo.shiftBy, camInfo.whyB, camInfo.costB, camInfo.zeroB, camInfo.Hb);
            }
            // Bilder + Werte speichern (fuer die Fehlersuche)
            std::string exportState = "FEHLGESCHLAGEN";
            try {
                const std::filesystem::path dir = DiagDir();
                std::filesystem::create_directories(dir);
                bool ok = WritePGM(dir / "aktuell.pgm", curLuma);
                if (haveF) ok = WritePGM(dir / "naechstes.pgm", lumF) && ok;
                if (haveB) ok = WritePGM(dir / "vorheriges.pgm", lumB) && ok;
                std::ofstream info(dir / "info.txt");
                for (const auto& l : L) info << l << "\n";
                info << "current_time=" << in_data->current_time << " time_step=" << in_data->time_step
                     << " local_time_step=" << in_data->local_time_step << " time_scale=" << in_data->time_scale
                     << " factor=" << factor << " haveF=" << haveF << " haveB=" << haveB << "\n";
                if (ok && info) exportState = "GESPEICHERT";
            } catch (...) {}
            L.push_back("EXPORT: " + exportState);
            diag::DrawLines(result, L);
        }
        WriteWorld(result, outKind, output, IsPremiere(in_data));
    } catch (const std::bad_alloc&) {
        return PF_Err_OUT_OF_MEMORY;
    } catch (...) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    (void)out_data;
    return err;
}

// ---------------------------------------------------------------------------
// Einstiegspunkte
// ---------------------------------------------------------------------------

extern "C" DllExport
PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                SPBasicSuite* inSPBasicSuitePtr, const char* inHostName,
                                const char* inHostVersion)
{
    (void)inSPBasicSuitePtr; (void)inHostName; (void)inHostVersion;
    PF_Err result = PF_Err_INVALID_CALLBACK;
    PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr,
                            FMB_NAME, FMB_MATCH_NAME, FMB_CATEGORY,
                            AE_RESERVED_INFO, "EffectMain", "");
    return result;
}

extern "C" DllExport
PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data,
                  PF_ParamDef* params[], PF_LayerDef* output, void* extra)
{
    (void)extra;
    PF_Err err = PF_Err_NONE;
    try {
        switch (cmd) {
        case PF_Cmd_ABOUT:        err = About(in_data, out_data); break;
        case PF_Cmd_GLOBAL_SETUP: err = GlobalSetup(in_data, out_data); break;
        case PF_Cmd_PARAMS_SETUP: err = ParamsSetup(in_data, out_data); break;
        case PF_Cmd_RENDER:       err = Render(in_data, out_data, params, output); break;
        default: break;
        }
    } catch (const std::bad_alloc&) {
        err = PF_Err_OUT_OF_MEMORY;
    } catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}
