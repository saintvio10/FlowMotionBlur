// test_core.cpp - Testet den Kern ohne Adobe-SDK.
// Erzeugt synthetische Bilder mit bekannter Bewegung, misst den Fehler
// des optischen Flusses und schreibt Ergebnisbilder als PPM.

#include "../src/core/FlowCore.h"
#include "../src/gpu/GpuBackend.h"

#include <algorithm>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

using namespace fmb;

static float Tex(float x, float y, float seed)
{
    float v = 0.5f;
    v += 0.18f * std::sin(x * 0.11f + seed) * std::cos(y * 0.07f + seed * 2);
    v += 0.12f * std::sin(x * 0.031f - y * 0.043f + seed * 3);
    v += 0.10f * std::sin((x + y) * 0.23f + seed);
    v += 0.08f * ((static_cast<int>(std::floor(x / 9)) + static_cast<int>(std::floor(y / 9))) & 1);
    return v;
}

static const int W = 640, H = 360;
static const float BG_DX = 4.f;                  // Hintergrund px/Frame
static const float FG_DX = 18.f, FG_DY = 6.f;    // Vordergrund px/Frame
static const float FG_X = 260, FG_Y = 130, FG_S = 100;

static bool InFG(float x, float y, float t)
{
    float px = FG_X + FG_DX * t, py = FG_Y + FG_DY * t;
    return x >= px && x < px + FG_S && y >= py && y < py + FG_S;
}

static void MakeFrame(float t, RGBAImage& img)
{
    img.resize(W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float* p = img.px(x, y);
            if (InFG(x, y, t)) {
                float px = FG_X + FG_DX * t, py = FG_Y + FG_DY * t;
                float v = Tex(x - px, y - py, 5.f);
                p[0] = v; p[1] = v * 0.3f; p[2] = 0.2f;
            } else {
                float v = Tex(x - BG_DX * t, y, 1.f);
                p[0] = v * 0.4f; p[1] = v * 0.7f; p[2] = v;
            }
            p[3] = 1.f;
        }
}

static void WritePPM(const std::string& path, const RGBAImage& img)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fprintf(f, "P6\n%d %d\n255\n", img.w, img.h);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) {
            const float* p = img.px(x, y);
            for (int c = 0; c < 3; ++c) {
                float v = p[c] < 0 ? 0 : (p[c] > 1 ? 1 : p[c]);
                std::fputc(static_cast<int>(v * 255.f + 0.5f), f);
            }
        }
    std::fclose(f);
}

// Mittlerer Endpunktfehler im Inneren von Vorder- bzw. Hintergrund.
static void Evaluate(const char* name, const Flow& fl, int scale, float sign)
{
    double eFG = 0, eBG = 0; int nFG = 0, nBG = 0;
    for (int y = 20; y < H - 20; y += 2)
        for (int x = 20; x < W - 20; x += 2) {
            // Randzonen (Verdeckung) auslassen
            bool fg = InFG(x, y, 0), fgIn = InFG(x - 12, y - 12, 0) && InFG(x + 12, y + 12, 0);
            bool nearFG = InFG(x - 30, y - 30, 0) || InFG(x + 30, y + 30, 0) ||
                          InFG(x - 30, y + 30, 0) || InFG(x + 30, y - 30, 0) || fg;
            int fx = x / scale, fy = y / scale;
            float u = fl.u.row(fy)[fx] * scale, v = fl.v.row(fy)[fx] * scale;
            if (fgIn) {
                eFG += std::hypot(u - sign * FG_DX, v - sign * FG_DY); ++nFG;
            } else if (!nearFG) {
                eBG += std::hypot(u - sign * BG_DX, v); ++nBG;
            }
        }
    std::printf("  %-10s Fehler Vordergrund: %.2f px   Hintergrund: %.2f px\n",
                name, eFG / nFG, eBG / nBG);
}

int main(int argc, char** argv)
{
    std::string out = argc > 1 ? argv[1] : ".";
    RGBAImage prev, cur, next;
    MakeFrame(-1, prev); MakeFrame(0, cur); MakeFrame(1, next);

    for (int scale : {1, 2, 4}) {
        auto t0 = std::chrono::steady_clock::now();
        Image l0, lp, ln;
        LumaFromRGBA(cur, scale, l0);
        LumaFromRGBA(prev, scale, lp);
        LumaFromRGBA(next, scale, ln);

        FlowParams fp;
        Flow fwd, bwd;
        ComputeFlow(l0, ln, fp, fwd);
        ComputeFlow(l0, lp, fp, bwd);

        BlurParams bp; bp.shutter = 0.5f; bp.maxSamples = 32;
        RGBAImage blurred;
        ApplyMotionBlur(cur, &fwd, &bwd, scale, bp, blurred);
        auto t1 = std::chrono::steady_clock::now();

        std::printf("Fluss-Aufloesung 1/%d  (%.0f ms fuer %dx%d)\n", scale,
                    std::chrono::duration<double, std::milli>(t1 - t0).count(), W, H);
        Evaluate("vorwaerts", fwd, scale, 1.f);
        Evaluate("rueckw.", bwd, scale, -1.f);

        WritePPM(out + "/blur_q" + std::to_string(scale) + ".ppm", blurred);
        if (scale == 2) {
            RGBAImage vis;
            VisualizeFlow(fwd, scale, W, H, vis);
            WritePPM(out + "/vectors.ppm", vis);
        }
    }
    WritePPM(out + "/original.ppm", cur);

    // ------------------------------------------------------------------
    // Track Points: sehr schnelle Bewegung, die das Tracking allein nicht
    // findet (Pyramide bewusst auf 2 Stufen begrenzt).
    // ------------------------------------------------------------------
    {
        Image l0, ln;
        LumaFromRGBA(cur, 1, l0);
        RGBAImage far; 
        // Vordergrund springt 18 px * 3 = 54 px
        MakeFrame(3, far);
        LumaFromRGBA(far, 1, ln);
        FlowParams fp; fp.maxLevels = 2;
        auto err = [&](const Flow& f) {
            double e = 0; int n = 0;
            for (int y = (int)FG_Y + 20; y < (int)(FG_Y + FG_S) - 20; y += 2)
                for (int x = (int)FG_X + 20; x < (int)(FG_X + FG_S) - 20; x += 2) {
                    e += std::hypot(f.u.row(y)[x] - 3 * FG_DX, f.v.row(y)[x] - 3 * FG_DY); ++n;
                }
            return e / n;
        };
        Flow plain, guided;
        ComputeFlow(l0, ln, fp, plain);
        std::vector<Guide> g(1);
        g[0].x = FG_X + FG_S / 2; g[0].y = FG_Y + FG_S / 2; g[0].dx = 3 * FG_DX; g[0].dy = 3 * FG_DY;
        ComputeFlow(l0, ln, fp, guided, &g);
        std::printf("Track Point  ohne: %.2f px   mit: %.2f px  (Fehler im Vordergrund)\n", err(plain), err(guided));
    }

    // ------------------------------------------------------------------
    // GPU gegen CPU
    // ------------------------------------------------------------------
    if (!gpu::Available()) {
        std::printf("GPU: kein OpenCL-Geraet - Vergleich uebersprungen\n");
    } else {
        std::printf("GPU: %s\n", gpu::DeviceName().c_str());
        const int scale = 2;
        Image l0, lp, ln;
        LumaFromRGBA(cur, scale, l0); LumaFromRGBA(prev, scale, lp); LumaFromRGBA(next, scale, ln);
        FlowParams fp;
        std::vector<Guide> g(1);
        g[0].x = 100; g[0].y = 100; g[0].dx = 2; g[0].dy = -1;

        Flow cf, cb, gf, gb;
        ComputeFlow(l0, ln, fp, cf, &g); ComputeFlow(l0, lp, fp, cb);
        auto t0 = std::chrono::steady_clock::now();
        bool ok = gpu::ComputeFlow(l0, ln, fp, gf, &g) && gpu::ComputeFlow(l0, lp, fp, gb);
        auto t1 = std::chrono::steady_clock::now();
        if (!ok) { std::printf("GPU-Fluss FEHLGESCHLAGEN\n"); return 1; }
        double du = 0, dmax = 0;
        for (size_t i = 0; i < cf.u.d.size(); ++i) {
            double d = std::hypot(cf.u.d[i] - gf.u.d[i], cf.v.d[i] - gf.v.d[i]);
            du += d; dmax = std::max(dmax, d);
        }
        std::printf("  Fluss  GPU vs CPU: mittl. Abweichung %.5f px, max %.4f px  (%.0f ms)\n",
                    du / cf.u.d.size(), dmax, std::chrono::duration<double, std::milli>(t1 - t0).count());

        BlurParams bp;
        RGBAImage cpuB, gpuB;
        ApplyMotionBlur(cur, &cf, &cb, scale, bp, cpuB);
        t0 = std::chrono::steady_clock::now();
        ok = gpu::ApplyMotionBlur(cur, &cf, &cb, scale, bp, gpuB);
        t1 = std::chrono::steady_clock::now();
        if (!ok) { std::printf("GPU-Blur FEHLGESCHLAGEN\n"); return 1; }
        double dm = 0, mx = 0;
        for (size_t i = 0; i < cpuB.d.size(); ++i) {
            double d = std::fabs(cpuB.d[i] - gpuB.d[i]); dm += d; mx = std::max(mx, d);
        }
        std::printf("  Blur   GPU vs CPU: mittl. Abweichung %.6f, max %.4f (Farbwerte 0..1)  (%.0f ms)\n",
                    dm / cpuB.d.size(), mx, std::chrono::duration<double, std::milli>(t1 - t0).count());

        // Mindestbewegung (Regler) auf CPU und GPU gleich
        {
            BlurParams bm; bm.minMotion = 3.f;
            RGBAImage c2, g2;
            ApplyMotionBlur(cur, &cf, &cb, scale, bm, c2);
            if (!gpu::ApplyMotionBlur(cur, &cf, &cb, scale, bm, g2)) { std::printf("GPU-Blur (Mindestbewegung) FEHLGESCHLAGEN\n"); return 1; }
            double m2 = 0; for (size_t i = 0; i < c2.d.size(); ++i) m2 = std::max(m2, (double)std::fabs(c2.d[i] - g2.d[i]));
            std::printf("  Blur Mindestbewegung 3 px: max. Abweichung GPU/CPU %.4f\n", m2);
        }
        // nur eine Richtung (Clip-Anfang/-Ende)
        ApplyMotionBlur(cur, &cf, nullptr, scale, bp, cpuB);
        gpu::ApplyMotionBlur(cur, &cf, nullptr, scale, bp, gpuB);
        mx = 0;
        for (size_t i = 0; i < cpuB.d.size(); ++i) mx = std::max(mx, (double)std::fabs(cpuB.d[i] - gpuB.d[i]));
        std::printf("  Blur nur vorwaerts: max. Abweichung %.4f\n", mx);
    }
    return 0;
}
