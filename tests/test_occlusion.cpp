// test_occlusion.cpp - Verdeckung: Kamera schwenkt (Hintergrund 4 px/Bild),
// ein Objekt bewegt sich schneller (18 px/Bild) und verdeckt/gibt Hintergrund frei.
// Vergleich mit/ohne Verdeckungskorrektur gegen die echte Belichtung.
#include "../src/core/FlowCore.h"
#include <cmath>
#include <cstdio>
using namespace fmb;
static const int W = 480, H = 320;
static float Tex(float x, float y, float s) { return 0.5f + 0.2f * std::sin(x * 0.17f + s) * std::cos(y * 0.13f) + 0.12f * std::sin((x + y) * 0.07f + 2 * s)
    + 0.08f * ((static_cast<int>(std::floor(x / 7)) + static_cast<int>(std::floor(y / 7))) & 1); }
static bool In(float x, float y, float t) { float px = 180 + 18 * t, py = 110; return x >= px && x < px + 110 && y >= py && y < py + 100; }
static void Frame(float t, RGBAImage& img)
{
    img.resize(W, H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        float* p = img.px(x, y); p[3] = 1;
        if (In(x + 0.5f, y + 0.5f, t)) { float v = Tex(x - 18 * t, y, 3); p[0] = v; p[1] = v * 0.4f; p[2] = 0.25f; }
        else { float v = Tex(x - 4 * t, y, 0); p[0] = v * 0.4f; p[1] = v * 0.7f; p[2] = v; }
    }
}
int main()
{
    RGBAImage prev, cur, next; Frame(-1, prev); Frame(0, cur); Frame(1, next);
    RGBAImage ref; ref.resize(W, H); const int N = 48;
    for (int i = 0; i < N; ++i) { RGBAImage s; Frame(-0.25f + 0.5f * (i + 0.5f) / N, s); for (size_t k = 0; k < ref.d.size(); ++k) ref.d[k] += s.d[k] / N; }
    const int scale = 2;
    Image l0, ln, lp; LumaFromRGBA(cur, scale, l0); LumaFromRGBA(next, scale, ln); LumaFromRGBA(prev, scale, lp);
    auto flowFn = [](const Image& a, const Image& b, const FlowParams& q, Flow& o) { ComputeFlow(a, b, q, o); };
    // Fehler nahe den Objektkanten (Verdeckungszonen) und gesamt
    auto err = [&](const RGBAImage& a, bool edges) {
        double e = 0; int n = 0;
        for (int y = 10; y < H - 10; ++y) for (int x = 10; x < W - 10; ++x) {
            bool nearEdge = In(x - 25, y, 0) != In(x + 25, y, 0) || In(x, y - 8, 0) != In(x, y + 8, 0);
            if (edges && !nearEdge) continue;
            for (int c = 0; c < 3; ++c) { e += std::fabs(a.px(x, y)[c] - ref.px(x, y)[c]); ++n; }
        }
        return e / n;
    };
    std::printf("Verdeckungs-Test (Hintergrund 4 px/Bild, Objekt 18 px/Bild)\n");
    std::printf("  ohne Unschaerfe:        Kanten %.4f   gesamt %.4f\n", err(cur, true), err(cur, false));
    for (int occ = 0; occ < 2; ++occ) {
        FlowParams fp; fp.occlusions = occ != 0;
        Flow f, b; CameraInfo ci;
        CameraFlows(l0, &ln, &lp, fp, flowFn, &f, &b, &ci);
        BlurParams bp; RGBAImage out; ApplyMotionBlur(cur, &f, &b, scale, bp, out);
        if (occ) { BlurParams bo; bo.mode = kBlurObjects; RGBAImage o2; ApplyMotionBlur(cur, &f, &b, scale, bo, o2);
                   std::printf("  Modus Objekte:            Kanten %.4f   gesamt %.4f\n", err(o2, true), err(o2, false)); }
        std::printf("  %s Kanten %.4f   gesamt %.4f   (korrigiert %.1f%%)\n", occ ? "mit Verdeckungskorrektur:" : "ohne Korrektur:         ",
                    err(out, true), err(out, false), ci.occluded * 100);
    }
    return 0;
}
