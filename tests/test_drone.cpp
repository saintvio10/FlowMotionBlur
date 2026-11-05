// test_drone.cpp - Drohnenflug-Szene: Bewegung nimmt nach unten zu (nah = schnell),
// dazu eine fast strukturlose Flaeche ("Fluss"). Misst Tracking-Fehler und
// schreibt Ergebnisbilder.
#include "../src/core/FlowCore.h"
#include <cmath>
#include <cstdio>
#include <string>
using namespace fmb;

static const int W = 360, H = 640;
static float Hash(int x, int y) { unsigned h = x * 374761393u + y * 668265263u; h = (h ^ (h >> 13)) * 1274126177u; return ((h ^ (h >> 16)) & 1023) / 1023.f; }
static float Speed(float y) { float a = y / H; return 1.f + 15.f * a * a; }      // px pro Frame (nach unten)
static bool River(float x) { return x > 140 && x < 230; }
static float Scene(float x, float ty, int ch)
{
    if (River(x)) return 0.20f + 0.004f * Hash((int)x, (int)std::floor(ty)) + (ch == 2 ? 0.05f : 0.f);
    float v = 0.45f + 0.18f * std::sin(x * 0.21f) * std::cos(ty * 0.17f) + 0.12f * std::sin((x - ty) * 0.09f)
            + 0.15f * (Hash((int)(x / 6), (int)std::floor(ty / 6)) - 0.5f);
    return ch == 0 ? v * 0.9f : (ch == 1 ? v : v * 0.6f);
}
// Bild t: Inhalt bewegt sich zeilenabhaengig nach unten
static void Frame(float t, RGBAImage& img)
{
    img.resize(W, H);
    for (int y = 0; y < H; ++y) {
        float ty = y - t * Speed(y);
        for (int x = 0; x < W; ++x) { float* p = img.px(x, y); for (int c = 0; c < 3; ++c) p[c] = Scene(x, ty, c); p[3] = 1; }
    }
}
static void WritePPM(const std::string& path, const RGBAImage& img)
{
    FILE* f = std::fopen(path.c_str(), "wb"); std::fprintf(f, "P6\n%d %d\n255\n", img.w, img.h);
    for (size_t i = 0; i < img.d.size(); i += 4) for (int c = 0; c < 3; ++c) { float v = img.d[i + c]; std::fputc((int)(255 * (v < 0 ? 0 : v > 1 ? 1 : v)), f); }
    std::fclose(f);
}
static void Eval(const char* name, const Flow& f, int scale)
{
    double eT = 0, eR = 0; int nT = 0, nR = 0;
    for (int y = 20; y < H - 20; y += 2) for (int x = 10; x < W - 10; x += 2) {
        int fx = x / scale, fy = y / scale;
        double e = std::hypot(f.u.row(fy)[fx] * scale - 0, f.v.row(fy)[fx] * scale - Speed(y));
        if (River(x) && River(x - 6) && River(x + 6)) { eR += e; ++nR; } else if (!River(x - 6) && !River(x + 6)) { eT += e; ++nT; }
    }
    std::printf("  %-26s Fehler Struktur: %5.2f px   glatte Flaeche (Fluss): %5.2f px\n", name, eT / nT, eR / nR);
}
int main(int argc, char** argv)
{
    std::string out = argc > 1 ? argv[1] : ".";
    RGBAImage prev, cur, next; Frame(-1, prev); Frame(0, cur); Frame(1, next);
    const int scale = 2;
    Image l0, lp, ln; LumaFromRGBA(cur, scale, l0); LumaFromRGBA(prev, scale, lp); LumaFromRGBA(next, scale, ln);
    FlowParams fp; Flow f, b; ComputeFlow(l0, ln, fp, f); ComputeFlow(l0, lp, fp, b);
    std::printf("Drohnen-Szene (Bewegung 1..16 px/Frame nach unten, Fluss = glatte Flaeche)\n");
    Eval("vorher (nur Tracking)", f, scale);
    Flow fr = f, br = b; RefineFlows(l0, &fr, &br);
    Eval("nachher (mit Auffuellen)", fr, scale);

    BlurParams bp; bp.shutter = 1.0f; bp.maxSamples = 48;
    RGBAImage oldB, newB;
    bp.mode = kBlurObjects; ApplyMotionBlur(cur, &f, &b, scale, bp, oldB);
    bp.mode = kBlurNatural; ApplyMotionBlur(cur, &fr, &br, scale, bp, newB);
    // Referenz: echte Belichtung = Mittel vieler Zwischenbilder
    RGBAImage ref; ref.resize(W, H); const int N = 64;
    for (int i = 0; i < N; ++i) { RGBAImage fr_; Frame(-0.5f + (i + 0.5f) / N, fr_); for (size_t k = 0; k < ref.d.size(); ++k) ref.d[k] += fr_.d[k] / N; }
    auto err = [&](const RGBAImage& a) { double e = 0; for (int y = 20; y < H - 20; ++y) for (int x = 0; x < W; ++x) for (int c = 0; c < 3; ++c) e += std::fabs(a.px(x, y)[c] - ref.px(x, y)[c]); return e / ((H - 40) * W * 3); };
    std::printf("  Abweichung zur echten Belichtung: alt %.4f   neu %.4f   (ohne Blur %.4f)\n", err(oldB), err(newB), err(cur));
    WritePPM(out + "/drone_orig.ppm", cur); WritePPM(out + "/drone_alt.ppm", oldB); WritePPM(out + "/drone_neu.ppm", newB); WritePPM(out + "/drone_ref.ppm", ref);
    return 0;
}
