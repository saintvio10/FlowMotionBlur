// PrGpuFilter.cpp - Premieres eigene GPU-Schnittstelle (nur macOS, Metal).
//
// Premiere sucht im Plugin nach "xGPUFilterEntry". Ist die Option
// "Premiere-GPU (Metal)" an, bleibt das Bild auf der Grafikkarte:
//   * Premiere liefert aktuelles Bild und Nachbarbilder als Metal-Puffer
//   * verkleinerte Luminanz wird auf der GPU berechnet (wenige MB zurueck)
//   * Bewegungsanalyse wie im Software-Weg auf der CPU
//   * Unschaerfe auf der GPU direkt in Premieres Ausgabebild
// Fuer alles andere (Vektor-Ebene, Track Points, Objekt-Modus, Vektor- und
// Diagnose-Ansicht) lehnt CreateInstance ab; Premiere nimmt dann automatisch
// den Software-Weg (FlowMotionBlur.cpp).
//
// Bewusst ohne PrGPUFilterModule.h aus dem SDK: Es braucht Header, die dem
// After-Effects-SDK fehlen. Verwendet werden nur die noetigen Suites.

#include "PrSDKGPUFilter.h"
#include "PrSDKGPUDeviceSuite.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKVideoSegmentSuite.h"
#include "PrSDKMemoryManagerSuite.h"
#include "PrSDKPlugSuites.h"
#include "PrSDKAESupport.h"
#include "SPBasic.h"

#include "FlowMotionBlur.h"
#include "../core/FlowCore.h"
#include "../gpu/GpuBackend.h"
#include "../gpu/MetalBackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Statusdatei: Die Diagnose-Ansicht (Software-Weg) zeigt daraus an, ob und wie
// schnell der GPU-Weg zuletzt gerechnet hat.
// ---------------------------------------------------------------------------
std::filesystem::path StatusFile()
{
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home ? home : ".") / "Library" / "Application Support" /
           "FlowMotionBlur" / "premiere_gpu.txt";
}

std::mutex gStatusMutex;
std::string gStatusInit = "-", gStatusRender = "-";
long gDeps = 0, gInstOk = 0, gInstNo = 0, gRenders = 0, gRendersOk = 0, gRendersFail = 0;
std::string gLastFail = "-";
std::map<std::string, int> gReasons; // Ablehnungsgruende mit Anzahl

// Zeile 1: Starts (aktiv/abgelehnt, letzter Grund). Zeile 2: Bilder und letztes Ergebnis.
void WriteStatus()
{
    try {
        const auto p = StatusFile();
        std::filesystem::create_directories(p.parent_path());
        std::ofstream f(p);
        std::string why;
        for (const auto& r : gReasons) why += (why.empty() ? "" : ", ") + r.first + " " + std::to_string(r.second);
        f << "STARTS: AKTIV " << gInstOk << "  ABGELEHNT " << gInstNo << (why.empty() ? "" : " (" + why + ")") << "\n"
          << "  ZULETZT: " << gStatusInit << "\n"
          << "ABFRAGEN NACHBARN: " << gDeps << "  BILDER: " << gRenders << "  OK " << gRendersOk << "  FEHLER " << gRendersFail
          << (gRendersFail ? "  LETZTER FEHLER: " + gLastFail : std::string()) << "\n"
          << "LETZTES: " << gStatusRender << "\n";
    } catch (...) {}
}
void SetInitStatus(const std::string& s, bool ok = false)
{
    std::lock_guard<std::mutex> l(gStatusMutex);
    (ok ? gInstOk : gInstNo)++;
    if (!ok && gReasons.size() < 12) ++gReasons[s];
    gStatusInit = s; WriteStatus();
}
// Wird am Ende jedes Render-Aufrufs gesetzt (Erfolg oder Grund des Abbruchs)
void SetRenderStatus(const std::string& s, bool fail = false)
{
    std::lock_guard<std::mutex> l(gStatusMutex);
    ++gRenders;
    if (fail) { ++gRendersFail; gLastFail = s; } else ++gRendersOk;
    gStatusRender = s; WriteStatus();
}

std::string Num(double v, int prec = 1)
{
    char b[64]; std::snprintf(b, sizeof(b), "%.*f", prec, v); return b;
}

double ParamNum(const PrParam& p)
{
    switch (p.mType) {
    case kPrParamType_Int8:    return p.mInt8;
    case kPrParamType_Int16:   return p.mInt16;
    case kPrParamType_Int32:   return p.mInt32;
    case kPrParamType_Int64:   return static_cast<double>(p.mInt64);
    case kPrParamType_Float32: return p.mFloat32;
    case kPrParamType_Float64: return p.mFloat64;
    case kPrParamType_Bool:    return p.mBool ? 1.0 : 0.0;
    default:                   return 0.0;
    }
}

void ResizeLuma(const fmb::Image& src, int w, int h, fmb::Image& dst)
{
    dst.resize(w, h);
    if (src.empty()) return;
    const float sx = static_cast<float>(src.w) / w, sy = static_cast<float>(src.h) / h;
    for (int y = 0; y < h; ++y) {
        const float fy = std::min<float>(src.h - 1.f, std::max(0.f, (y + 0.5f) * sy - 0.5f));
        const int y0 = static_cast<int>(fy), y1 = std::min(y0 + 1, src.h - 1);
        const float ty = fy - y0;
        for (int x = 0; x < w; ++x) {
            const float fx = std::min<float>(src.w - 1.f, std::max(0.f, (x + 0.5f) * sx - 0.5f));
            const int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, src.w - 1);
            const float tx = fx - x0;
            const float a = src.row(y0)[x0] + (src.row(y0)[x1] - src.row(y0)[x0]) * tx;
            const float b = src.row(y1)[x0] + (src.row(y1)[x1] - src.row(y1)[x0]) * tx;
            dst.row(y)[x] = a + (b - a) * ty;
        }
    }
}

double MeanAbsDiff(const fmb::Image& a, const fmb::Image& b)
{
    if (a.d.size() != b.d.size() || a.d.empty()) return 1.0;
    double d = 0;
    for (size_t i = 0; i < a.d.size(); ++i) d += std::fabs(a.d[i] - b.d[i]);
    return d / a.d.size();
}

// Reihenfolge der angeforderten Nachbarbilder (inFrames[1..4])
const int kDepOffsets[4] = { -1, +1, -2, +2 };

class FlowMotionBlurGPU {
public:
    prSuiteError Initialize(PrGPUFilterInstance* inst)
    {
        mNodeID = inst->inNodeID;
        SPBasicSuite* sp = inst->piSuites && inst->piSuites->utilFuncs ? inst->piSuites->utilFuncs->getSPBasicSuite() : nullptr;
        if (!sp) return suiteError_Fail;
        mSP = sp;
        if (sp->AcquireSuite(kPrSDKGPUDeviceSuite, kPrSDKGPUDeviceSuiteVersion, (const void**)&mDevice) != kSPNoError ||
            sp->AcquireSuite(kPrSDKPPixSuite, kPrSDKPPixSuiteVersion, (const void**)&mPPix) != kSPNoError ||
            sp->AcquireSuite(kPrSDKVideoSegmentSuite, kPrSDKVideoSegmentSuiteVersion, (const void**)&mSegment) != kSPNoError ||
            !mDevice || !mPPix || !mSegment) {
            SetInitStatus("SUITES FEHLEN");
            return suiteError_Fail;
        }
        mDeviceIndex = inst->inDeviceIndex;
        PrGPUDeviceInfo info = {};
        if (mDevice->GetDeviceInfo(kPrSDKGPUDeviceSuiteVersion, mDeviceIndex, &info) != suiteError_NoError) {
            SetInitStatus("GERAET UNBEKANNT");
            return suiteError_Fail;
        }
        if (info.outDeviceFramework != PrGPUDeviceFramework_Metal) {
            SetInitStatus("KEIN METAL");
            return suiteError_Fail;
        }
        mMtlDevice = info.outDeviceHandle;
        mMtlQueue = info.outCommandQueueHandle;
        if (!mMtlDevice || !mMtlQueue) { SetInitStatus("KEIN METAL-GERAET"); return suiteError_Fail; }

        // Nur der einfache, haeufige Fall laeuft auf Premieres GPU-Weg.
        const PrTime t0 = 0;
        if (Num0(FMB_PR_GPU, t0) == 0) { SetInitStatus("OPTION AUS"); return suiteError_Fail; }
        if (Pop(FMB_VIEW, t0) != VIEW_RESULT) { SetInitStatus("ANSICHT"); return suiteError_Fail; }
        if (Pop(FMB_BLUR_MODE, t0) != MODE_NATURAL) { SetInitStatus("OBJEKT-MODUS"); return suiteError_Fail; }
        if (Pop(FMB_VECTOR_SOURCE, t0) != SOURCE_TRACKING) { SetInitStatus("VEKTOR-EBENE"); return suiteError_Fail; }
        // Track Points: Ob Premiere hier die Gruppen-Anfaenge/-Enden mitzaehlt, ist nicht
        // dokumentiert. Daher beide Zaehlweisen pruefen (3 Gruppenmarken davor);
        // gewertet wird nur ein echter Schalter (Bool), der an ist.
        for (int i = 0; i < FMB_NUM_TRACK_POINTS; ++i) {
            for (int shift : { 0, 3 }) {
                PrParam p = Get(FMB_TP_ON(i) - shift, t0);
                if (p.mType == kPrParamType_Bool && p.mBool) {
                    // Nachbar muss ein Punkt sein, sonst ist es nicht der Track-Point-Schalter
                    PrParam pos = Get(FMB_TP_POS(i) - shift, t0);
                    if (pos.mType == kPrParamType_Point) { SetInitStatus("TRACK POINT"); return suiteError_Fail; }
                }
            }
        }
        SetInitStatus("AKTIV (METAL)", true);
        return suiteError_NoError;
    }

    prSuiteError GetFrameDependencies(const PrGPUFilterRenderParams* rp, csSDK_int32* ioQueryIndex,
                                      PrGPUFilterFrameDependency* out)
    {
        const int qi = ioQueryIndex ? *ioQueryIndex : 0;
        if (qi == 0) { std::lock_guard<std::mutex> l(gStatusMutex); ++gDeps; WriteStatus(); }
        if (qi < 0 || qi >= 4) return suiteError_InvalidParms;
        out->outDependencyType = PrGPUDependency_InputFrame;
        out->outTrackID = 0;
        out->outSequenceTime = rp->inSequenceTime + kDepOffsets[qi] * rp->inRenderTicksPerFrame;
        if (ioQueryIndex && qi + 1 < 4) *ioQueryIndex = qi + 1;
        return suiteError_NoError;
    }

    prSuiteError Render(const PrGPUFilterRenderParams* rp, const PPixHand* inFrames, csSDK_size_t inFrameCount,
                        PPixHand* outFrame)
    {
        try {
            return RenderImpl(rp, inFrames, inFrameCount, outFrame);
        } catch (...) {
            SetRenderStatus("AUSNAHME", true);
            return suiteError_Fail;
        }
    }

    ~FlowMotionBlurGPU()
    {
        if (mSP) {
            if (mDevice) mSP->ReleaseSuite(kPrSDKGPUDeviceSuite, kPrSDKGPUDeviceSuiteVersion);
            if (mPPix) mSP->ReleaseSuite(kPrSDKPPixSuite, kPrSDKPPixSuiteVersion);
            if (mSegment) mSP->ReleaseSuite(kPrSDKVideoSegmentSuite, kPrSDKVideoSegmentSuiteVersion);
        }
    }

private:
    PrParam Get(int index, PrTime t)
    {
        PrParam p = {};
        // Premieres GPU-Parameter zaehlen ohne das Eingabebild
        mSegment->GetParam(mNodeID, index - 1, t, &p);
        return p;
    }
    double Num0(int index, PrTime t) { return ParamNum(Get(index, t)); }
    // Popups sind hier 0-basiert, im Software-Weg 1-basiert
    int Pop(int index, PrTime t) { return static_cast<int>(std::lround(ParamNum(Get(index, t)))) + 1; }

    bool FrameOf(PPixHand h, fmb::metal::Frame& f, PrPixelFormat& fmt)
    {
        if (!h) return false;
        if (mPPix->GetPixelFormat(h, &fmt) != suiteError_NoError) return false;
        if (fmt != PrPixelFormat_GPU_BGRA_4444_32f && fmt != PrPixelFormat_GPU_BGRA_4444_16f) return false;
        prRect r = {};
        csSDK_int32 rb = 0;
        void* data = nullptr;
        if (mPPix->GetBounds(h, &r) != suiteError_NoError) return false;
        if (mPPix->GetRowBytes(h, &rb) != suiteError_NoError || rb <= 0) return false;
        if (mDevice->GetGPUPPixData(h, &data) != suiteError_NoError || !data) return false;
        f.buffer = data;
        f.w = r.right - r.left; f.h = r.bottom - r.top;
        f.rowBytes = rb;
        f.isHalf = fmt == PrPixelFormat_GPU_BGRA_4444_16f;
        return f.w > 0 && f.h > 0 && rb >= f.w * (f.isHalf ? 8 : 16);
    }

    prSuiteError RenderImpl(const PrGPUFilterRenderParams* rp, const PPixHand* inFrames, csSDK_size_t inFrameCount,
                            PPixHand* outFrame)
    {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        if (!inFrames || inFrameCount < 1 || !outFrame) { SetRenderStatus("KEINE EINGABE", true); return suiteError_Fail; }

        const PrTime ct = rp->inClipTime;
        const float amount = static_cast<float>(Num0(FMB_AMOUNT, ct));
        const int samples = static_cast<int>(std::lround(Num0(FMB_MAX_SAMPLES, ct)));
        const int quality = Pop(FMB_QUALITY, ct);
        const int smoothing = static_cast<int>(std::lround(Num0(FMB_SMOOTHING, ct)));
        const bool rgbaTrack = Num0(FMB_RGBA_TRACKING, ct) != 0;
        const bool useCL = Num0(FMB_USE_GPU, ct) != 0;
        const bool cameraMode = Pop(FMB_MOTION_TYPE, ct) != MOTION_FREE;
        const float alphaWeight = rgbaTrack ? 0.5f : 0.f;

        // Ausgabe: Premiere rendert "an Ort und Stelle" - outFrame enthaelt schon
        // das aktuelle Bild. Fehlt es, eine Kopie des Eingabebildes anlegen.
        fmb::metal::Frame cur;
        PrPixelFormat fmt = PrPixelFormat_Invalid;
        if (!FrameOf(inFrames[0], cur, fmt)) { SetRenderStatus("FORMAT NICHT UNTERSTUETZT", true); return suiteError_Fail; }
        if (!*outFrame) {
            csSDK_uint32 parN = 1, parD = 1;
            mPPix->GetPixelAspectRatio(inFrames[0], &parN, &parD);
            if (mDevice->CreateGPUPPix(mDeviceIndex, fmt, cur.w, cur.h, parN, parD, prFieldsNone, outFrame) != suiteError_NoError) {
                SetRenderStatus("AUSGABE ANLEGEN", true); return suiteError_Fail;
            }
            fmb::metal::Frame o; PrPixelFormat f2 = PrPixelFormat_Invalid;
            if (!FrameOf(*outFrame, o, f2) || !fmb::metal::Copy(mMtlDevice, mMtlQueue, cur, o)) {
                SetRenderStatus("AUSGABE KOPIEREN", true); return suiteError_Fail;
            }
        }
        fmb::metal::Frame out;
        PrPixelFormat outFmt = PrPixelFormat_Invalid;
        if (!FrameOf(*outFrame, out, outFmt) || out.w != cur.w || out.h != cur.h) {
            SetRenderStatus("AUSGABE UNPASSEND", true); return suiteError_Fail;
        }
        if (amount <= 0.f) { SetRenderStatus("STAERKE 0"); return suiteError_NoError; }

        // Vorschau-Verkleinerung (Premiere gibt 0.5 oder 2 an - beides zulassen)
        double ds = rp->inDownsampleFactorX;
        if (!(ds > 0) || !std::isfinite(ds)) ds = 1;
        if (ds < 1) ds = 1 / ds;

        int factor = (quality == QUALITY_DRAFT) ? 4 : (quality == QUALITY_HIGH ? 1 : 2);
        factor = std::max(1, factor / std::max(1, static_cast<int>(ds + 0.5)));

        fmb::BlurParams bp;
        bp.shutter = amount;
        bp.maxSamples = std::max(3, samples);
        bp.minMotion = static_cast<float>(Num0(FMB_MIN_MOTION, ct) / ds);
        bp.mode = fmb::kBlurNatural;

        fmb::FlowParams fp;
        fp.smoothRadius = smoothing;
        fp.windowRadius = (factor >= 4) ? 2 : 3;

        // Luminanz aller Bilder auf der GPU
        fmb::Image curLuma;
        double curAlpha = 0;
        if (!fmb::metal::Luma(mMtlDevice, mMtlQueue, cur, factor, alphaWeight, curLuma, curAlpha)) {
            SetRenderStatus("METAL: " + fmb::metal::LastError(), true); return suiteError_Fail;
        }
        struct Nb { bool have = false; fmb::Image luma; double diff = 1; };
        Nb nb[4];
        for (int i = 0; i < 4; ++i) {
            if (static_cast<csSDK_size_t>(i + 1) >= inFrameCount) break;
            fmb::metal::Frame f; PrPixelFormat ff = PrPixelFormat_Invalid;
            if (!FrameOf(inFrames[i + 1], f, ff)) continue;
            double a = 0;
            fmb::Image l;
            const int f2 = (f.w == cur.w) ? factor : std::max(1, static_cast<int>(std::lround(factor * static_cast<double>(f.w) / cur.w)));
            if (!fmb::metal::Luma(mMtlDevice, mMtlQueue, f, f2, alphaWeight, l, a)) continue;
            // Ausserhalb des Clips: leeres (transparentes) Bild
            if (curAlpha > 0.0 && a <= 0.02 * curAlpha) continue;
            if (l.w != curLuma.w || l.h != curLuma.h) ResizeLuma(l, curLuma.w, curLuma.h, nb[i].luma);
            else nb[i].luma = std::move(l);
            nb[i].have = true;
            nb[i].diff = MeanAbsDiff(nb[i].luma, curLuma);
        }
        const auto tLuma = Clock::now();

        // Doppelte Bilder: gleicht der direkte Nachbar dem aktuellen Bild, den
        // uebernaechsten nehmen und die Bewegung halbieren.
        const double kDup = 0.0025;
        auto pick = [&](int n1, int n2, int& k) -> Nb* {
            k = 1;
            if (!nb[n1].have) return nullptr;
            if (nb[n1].diff < kDup && nb[n2].have && nb[n2].diff >= kDup) { k = 2; return &nb[n2]; }
            return &nb[n1];
        };
        int kB = 1, kF = 1;
        Nb* pB = pick(0, 2, kB);
        Nb* pF = pick(1, 3, kF);
        const int rep = std::max(kF, kB);

        // Falsche Nachbarbilder (Zeitverzerrung): vorheriges und naechstes aehneln
        // sich, weichen aber beide stark vom aktuellen ab -> nicht verwischen.
        if (pF && pB) {
            const double dNP = MeanAbsDiff(pF->luma, pB->luma);
            const double dCur = std::min(pF->diff, pB->diff);
            if (dCur > 0.06 && dNP < 0.4 * dCur) {
                SetRenderStatus("FALSCHE NACHBARBILDER (ZEITVERZERRUNG) - UNVERAENDERT");
                return suiteError_NoError;
            }
        }
        if (!pF && !pB) { SetRenderStatus("KEINE NACHBARBILDER"); return suiteError_NoError; }

        auto flowFn = [&](const fmb::Image& a, const fmb::Image& b, const fmb::FlowParams& q, fmb::Flow& o) {
            if (useCL && fmb::gpu::ComputeFlow(a, b, q, o, nullptr)) return;
            fmb::ComputeFlow(a, b, q, o, nullptr);
        };
        auto scaleFlow = [](fmb::Flow& f, int k) {
            if (k <= 1) return;
            const float inv = 1.f / k;
            for (float& v : f.u.d) v *= inv;
            for (float& v : f.v.d) v *= inv;
        };

        fmb::Flow fwd, bwd;
        std::string what;
        if (cameraMode) {
            fmb::CameraPrior prior; bool usePrior = false;
            const PrTime step = rp->inRenderTicksPerFrame;
            {
                std::lock_guard<std::mutex> l(mMutex);
                if (mHavePrior && std::llabs(rp->inSequenceTime - mPriorTime) <= 3 * step * rep) { prior = mPrior; usePrior = true; }
            }
            fmb::CameraInfo ci;
            fmb::CameraFlows(curLuma, pF ? &pF->luma : nullptr, pB ? &pB->luma : nullptr, fp, flowFn,
                             pF ? &fwd : nullptr, pB ? &bwd : nullptr, &ci, usePrior ? &prior : nullptr);
            {
                std::lock_guard<std::mutex> l(mMutex);
                mPrior.validF = ci.modelF; mPrior.validB = ci.modelB;
                std::copy(ci.Hf, ci.Hf + 9, mPrior.Hf); std::copy(ci.Hb, ci.Hb + 9, mPrior.Hb);
                mPriorTime = rp->inSequenceTime; mHavePrior = ci.modelF || ci.modelB;
            }
            what = "KAMERA MODELL " + Num(ci.sel.model * 100, 0) + "%";
        } else {
            auto one = [&](Nb* p, fmb::Flow& f) {
                fmb::FlowParams q = fp;
                fmb::EstimateGlobalShift(curLuma, p->luma, q.initU, q.initV);
                flowFn(curLuma, p->luma, q, f);
            };
            if (pF) one(pF, fwd);
            if (pB) one(pB, bwd);
            what = "FREI";
        }
        if (pF) scaleFlow(fwd, rep);
        if (pB) scaleFlow(bwd, rep);
        if (!cameraMode) fmb::RefineFlows(curLuma, pF ? &fwd : nullptr, pB ? &bwd : nullptr);
        const auto tFlow = Clock::now();

        if (!fmb::metal::BlurNaturalInPlace(mMtlDevice, mMtlQueue, out, pF ? &fwd : nullptr, pB ? &bwd : nullptr,
                                       factor, bp)) {
            SetRenderStatus("METAL: " + fmb::metal::LastError(), true); return suiteError_Fail;
        }
        const auto tEnd = Clock::now();
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::to_string(static_cast<int>(std::chrono::duration<double, std::milli>(b - a).count()));
        };
        SetRenderStatus(std::to_string(cur.w) + "X" + std::to_string(cur.h) + (cur.isHalf ? " 16F" : " 32F") +
                        "  " + std::to_string(inFrameCount) + " EIN  " + what + (rep > 1 ? "  DOPPELT" : "") +
                        "  MS "  + ms(t0, tLuma) + " / " + ms(tLuma, tFlow) + " / " + ms(tFlow, tEnd));
        return suiteError_NoError;
    }

    SPBasicSuite* mSP = nullptr;
    PrSDKGPUDeviceSuite* mDevice = nullptr;
    PrSDKPPixSuite* mPPix = nullptr;
    PrSDKVideoSegmentSuite* mSegment = nullptr;
    csSDK_int32 mNodeID = 0;
    csSDK_uint32 mDeviceIndex = 0;
    void* mMtlDevice = nullptr;
    void* mMtlQueue = nullptr;

    std::mutex mMutex;
    fmb::CameraPrior mPrior;
    PrTime mPriorTime = 0;
    bool mHavePrior = false;
};

// --- Funktionstabelle fuer Premiere ---------------------------------------
prSuiteError CreateInstance(PrGPUFilterInstance* inst)
{
    FlowMotionBlurGPU* f = new (std::nothrow) FlowMotionBlurGPU();
    if (!f) return suiteError_OutOfMemory;
    prSuiteError r = suiteError_Fail;
    try { r = f->Initialize(inst); } catch (...) { r = suiteError_Fail; }
    if (r == suiteError_NoError) { inst->ioPrivatePluginData = f; inst->outIsRealtime = kPrFalse; }
    else delete f;
    return r;
}
prSuiteError DisposeInstance(PrGPUFilterInstance* inst)
{
    delete static_cast<FlowMotionBlurGPU*>(inst->ioPrivatePluginData);
    inst->ioPrivatePluginData = nullptr;
    return suiteError_NoError;
}
prSuiteError GetFrameDependencies(PrGPUFilterInstance* inst, const PrGPUFilterRenderParams* rp,
                                  csSDK_int32* ioQueryIndex, PrGPUFilterFrameDependency* out)
{
    return static_cast<FlowMotionBlurGPU*>(inst->ioPrivatePluginData)->GetFrameDependencies(rp, ioQueryIndex, out);
}
prSuiteError Precompute(PrGPUFilterInstance*, const PrGPUFilterRenderParams*, csSDK_int32, PPixHand)
{
    return suiteError_NotImplemented;
}
prSuiteError Render(PrGPUFilterInstance* inst, const PrGPUFilterRenderParams* rp, const PPixHand* inFrames,
                    csSDK_size_t inFrameCount, PPixHand* outFrame)
{
    return static_cast<FlowMotionBlurGPU*>(inst->ioPrivatePluginData)->Render(rp, inFrames, inFrameCount, outFrame);
}

} // namespace

extern "C" __attribute__((visibility("default")))
prSuiteError xGPUFilterEntry(csSDK_uint32 inHostInterfaceVersion, csSDK_int32* ioIndex, prBool inStartup,
                             piSuitesPtr piSuites, PrGPUFilter* outFilter, PrGPUFilterInfo* outFilterInfo)
{
    (void)inHostInterfaceVersion; (void)piSuites;
    if (!inStartup) return suiteError_NoError;
    if (!ioIndex || !outFilter || !outFilterInfo) return suiteError_InvalidParms;
    if (*ioIndex != 0) return suiteError_InvalidParms; // genau ein Effekt
    outFilter->CreateInstance = CreateInstance;
    outFilter->DisposeInstance = DisposeInstance;
    outFilter->GetFrameDependencies = GetFrameDependencies;
    outFilter->Precompute = Precompute;
    outFilter->Render = Render;
    outFilterInfo->outMatchName = PrSDKString(); // leer = Name aus der PiPL
    outFilterInfo->outInterfaceVersion = PrSDKGPUFilterInterfaceVersion;
    return suiteError_NoError;
}
