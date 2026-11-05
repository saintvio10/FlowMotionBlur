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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <new>

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

static PixKind GetPixKind(PF_InData* in_data, PF_EffectWorld* world)
{
    PF_PixelFormat fmt = PF_PixelFormat_INVALID;
    {
        Suite<PF_WorldSuite2> ws(in_data, kPFWorldSuite, kPFWorldSuiteVersion2);
        if (ws) ws->PF_GetPixelFormat(world, &fmt);
    }
    const int f = static_cast<int>(fmt);

    // Premiere liefert neben den angemeldeten BGRA-Formaten immer auch sein
    // Standardformat ARGB 8 Bit (gleiches Layout wie in After Effects).
    if (f == static_cast<int>(PrPixelFormat_BGRA_4444_8u))  return PixKind::PR_BGRA8;
    if (f == static_cast<int>(PrPixelFormat_BGRA_4444_32f)) return PixKind::PR_BGRA32F;
    if (f == static_cast<int>(PF_PixelFormat_ARGB128)) return PixKind::AE_ARGB32F;
    if (f == static_cast<int>(PF_PixelFormat_ARGB64))  return PixKind::AE_ARGB16;
    if (f == static_cast<int>(PF_PixelFormat_ARGB32))  return PixKind::AE_ARGB8;
    return PF_WORLD_IS_DEEP(world) ? PixKind::AE_ARGB16 : PixKind::AE_ARGB8;
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
        // Premiere: nur BGRA 8 Bit und 32 Bit Float anbieten.
        Suite<PF_PixelFormatSuite1> pfs(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1);
        if (pfs) {
            pfs->ClearSupportedPixelFormats(in_data->effect_ref);
            pfs->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
            pfs->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
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
    PF_ADD_POPUP("Analyse-Qualitaet", 3, QUALITY_NORMAL,
                 "Schnell (1/4)|Normal (1/2)|Hoch (voll)", QUALITY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER("Vektor-Glaettung", 0, 10, 0, 10, 2, SMOOTHING_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("RGB+A-Tracking", FALSE, 0, RGBA_TRACKING_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("GPU verwenden", TRUE, 0, USE_GPU_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Ansicht", 2, VIEW_RESULT, "Ergebnis|Bewegungsvektoren", VIEW_DISK_ID);

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
    void Flow(const fmb::Image& a, const fmb::Image& b, const fmb::FlowParams& fp,
              const std::vector<fmb::Guide>* g, fmb::Flow& out) const
    {
        if (useGpu && fmb::gpu::ComputeFlow(a, b, fp, out, g)) return;
        fmb::ComputeFlow(a, b, fp, out, g);
    }
    void Blur(const fmb::RGBAImage& src, const fmb::Flow* f, const fmb::Flow* b, int scale,
              const fmb::BlurParams& bp, fmb::RGBAImage& dst) const
    {
        if (useGpu && fmb::gpu::ApplyMotionBlur(src, f, b, scale, bp, dst)) return;
        fmb::ApplyMotionBlur(src, f, b, scale, bp, dst);
    }
};

// Holt den Frame bei current_time + offset und berechnet den Fluss dorthin.
// Gibt false zurueck, wenn kein brauchbarer Nachbarframe existiert (Clip-Anfang/-Ende).
static bool NeighbourFlow(PF_InData* in_data, A_long offset, const fmb::RGBAImage& cur,
                          const fmb::Image& curLuma, int factor, const fmb::FlowParams& fp,
                          float alphaWeight, double curAlpha, const std::vector<fmb::Guide>& guides,
                          const Engine& eng, fmb::Flow& flow, PF_Err& err)
{
    PF_ParamDef ck;
    AEFX_CLR_STRUCT(ck);
    err = PF_CHECKOUT_PARAM(in_data, FMB_INPUT, in_data->current_time + offset,
                            in_data->time_step, in_data->time_scale, &ck);
    if (err != PF_Err_NONE) {
        // Abbruch durch den Benutzer weitergeben, alles andere = kein Nachbarframe
        if (err != PF_Interrupt_CANCEL) err = PF_Err_NONE;
        return false;
    }

    bool ok = false;
    PF_EffectWorld* w = &ck.u.ld;
    if (w->data && w->width == cur.w && w->height == cur.h) {
        PixKind kind = GetPixKind(in_data, w);
        if (kind != PixKind::Unknown) {
            fmb::RGBAImage nb;
            ReadWorld(w, kind, nb, IsPremiere(in_data));
            // Ausserhalb des Clips liefern Hosts oft ein leeres (transparentes) Bild.
            if (curAlpha <= 0.0 || AlphaSum(nb) > 0.02 * curAlpha) {
                fmb::Image nbLuma;
                fmb::LumaFromRGBA(nb, factor, nbLuma, alphaWeight);
                eng.Flow(curLuma, nbLuma, fp, guides.empty() ? nullptr : &guides, flow);
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

        A_long step = in_data->time_step != 0 ? in_data->time_step : in_data->local_time_step;
        if (step == 0) step = 1;

        const std::vector<fmb::Guide> gF = CollectGuides(in_data, params, +step, factor);
        const std::vector<fmb::Guide> gB = CollectGuides(in_data, params, -step, factor);

        fmb::Flow fwd, bwd;
        bool haveF = NeighbourFlow(in_data, +step, cur, curLuma, factor, fp, alphaWeight, curAlpha, gF, eng, fwd, err);
        if (err) return err;
        if ((err = PF_ABORT(in_data)) != PF_Err_NONE) return err;
        bool haveB = NeighbourFlow(in_data, -step, cur, curLuma, factor, fp, alphaWeight, curAlpha, gB, eng, bwd, err);
        if (err) return err;
        if ((err = PF_ABORT(in_data)) != PF_Err_NONE) return err;

        if (view == VIEW_VECTORS) {
            const fmb::Flow& show = haveF ? fwd : bwd;
            fmb::VisualizeFlow(show, factor, cur.w, cur.h, result);
        } else {
            eng.Blur(cur, haveF ? &fwd : nullptr, haveB ? &bwd : nullptr, factor, bp, result);
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
