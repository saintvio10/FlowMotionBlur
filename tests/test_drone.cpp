// test_drone.cpp - Physikalisch korrekte Drohnen-Szene: Kamera schaut schraeg nach
// unten und fliegt vorwaerts ueber eine Landschaft (Boden-Ebene mit Textur,
// glatter Fluss, Himmel ueber dem Horizont). Die echte Bewegung ist bekannt.
// Gemessen: Tracking-Fehler, Abweichung zur echten Belichtung, Flackern.
#include "../src/core/FlowCore.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace fmb;

static const int W = 360, H = 640;
static const double F = 380, PITCH = 0.42, HEIGHT = 40.0; // Brennweite px, Neigung rad, Flughoehe
struct V3 { double x, y, z; };
static V3 Ray(double px, double py)
{
    double xr = (px - W / 2.0) / F, yr = (py - H / 2.0) / F;
    double c = std::cos(PITCH), s = std::sin(PITCH);
    // rechts=(1,0,0), unten=(0,-s,-c), vorne=(0,c,-s)
    return { xr, -yr * s + c, -yr * c - s };
}
static double Hash(int x, int y) { unsigned h = x * 374761393u + y * 668265263u; h = (h ^ (h >> 13)) * 1274126177u; return ((h ^ (h >> 16)) & 1023) / 1023.0; }
static void Ground(double X, double Y, float out[3])
{
    if (X > -14 && X < -4) { float n = 0.004f * (float)Hash((int)std::floor(X * 4), (int)std::floor(Y * 4)); out[0] = 0.18f + n; out[1] = 0.20f + n; out[2] = 0.24f + n; return; }
    double v = 0.45 + 0.15 * std::sin(X * 0.9) * std::cos(Y * 0.7) + 0.2 * (Hash((int)std::floor(X / 1.5), (int)std::floor(Y / 1.5)) - 0.5)
             + 0.08 * std::sin((X + Y) * 2.3);
    bool house = std::fmod(std::fabs(X - 6), 9.0) < 4.0 && std::fmod(std::fabs(Y), 11.0) < 5.0 && X > 2;
    if (house) { out[0] = (float)(0.6 + 0.2 * v); out[1] = (float)(0.35 + 0.1 * v); out[2] = (float)(0.3 + 0.1 * v); return; }
    out[0] = (float)(0.55 * v); out[1] = (float)(0.75 * v); out[2] = (float)(0.35 * v);
}
// Bild zur Kameraposition camY (vorwaerts)
static void Render(double camY, RGBAImage& img)
{
    img.resize(W, H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        float* p = img.px(x, y); p[3] = 1;
        V3 d = Ray(x + 0.5, y + 0.5);
        if (d.z > -0.02) { double az = d.x / std::max(0.05, d.y), el = d.z; float c = (float)(0.6 + 0.2 * std::sin(az * 9) * std::cos(el * 20) + 0.1 * std::sin(az * 23 + el * 11));
            p[0] = c * 0.85f; p[1] = c * 0.9f; p[2] = c; continue; }
        double t = HEIGHT / -d.z; Ground(t * d.x, camY + t * d.y, p);
    }
}
// Echte Bewegung eines Pixels bei Vorwaertsflug um dY
static void TrueFlow(double px, double py, double dY, double& u, double& v)
{
    V3 d = Ray(px, py);
    if (d.z > -0.02) { u = v = 0; return; } // Himmel: unendlich weit
    double t = HEIGHT / -d.z; V3 G = { t * d.x, t * d.y - dY, -HEIGHT };
    double c = std::cos(PITCH), s = std::sin(PITCH);
    double Xc = G.x, Yc = -(G.y * s + G.z * c) * -1.0; // unten-Achse
    Yc = G.y * -s + G.z * -c; double Zc = G.y * c + G.z * -s;
    u = W / 2.0 + F * Xc / Zc - px; v = H / 2.0 + F * Yc / Zc - py;
}
static void WritePPM(const std::string& path, const RGBAImage& img)
{
    FILE* f = std::fopen(path.c_str(), "wb"); std::fprintf(f, "P6\n%d %d\n255\n", img.w, img.h);
    for (size_t i = 0; i < img.d.size(); i += 4) for (int c = 0; c < 3; ++c) { float v = img.d[i + c]; std::fputc((int)(255 * (v < 0 ? 0 : v > 1 ? 1 : v)), f); }
    std::fclose(f);
}

enum Method { OLD, REFINE, CAMERA };
struct Res { double epe, blurErr; RGBAImage blurred; };

static Res Run(double camY, double speed, Method m)
{
    const int scale = 2;
    RGBAImage prev, cur, next; Render(camY - speed, prev); Render(camY, cur); Render(camY + speed, next);
    Image l0, lp, ln; LumaFromRGBA(cur, scale, l0); LumaFromRGBA(prev, scale, lp); LumaFromRGBA(next, scale, ln);
    FlowParams fp; Flow f, b;
    if (m == OLD) { ComputeFlow(l0, ln, fp, f); ComputeFlow(l0, lp, fp, b); }
    else {
        FlowParams ff = fp, fb = fp;
        EstimateGlobalShift(l0, ln, ff.initU, ff.initV); EstimateGlobalShift(l0, lp, fb.initU, fb.initV);
        ComputeFlow(l0, ln, ff, f); ComputeFlow(l0, lp, fb, b);
        if (m == CAMERA) { if (!FitCameraModel(l0, &f, &b)) RefineFlows(l0, &f, &b); }
        else RefineFlows(l0, &f, &b);
    }
    // Tracking-Fehler (Boden, nicht Himmel, Rand 10 px)
    double e = 0; int n = 0;
    for (int y = 10; y < H - 10; y += 3) for (int x = 10; x < W - 10; x += 3) {
        double tu, tv; TrueFlow(x + 0.5, y + 0.5, speed, tu, tv);
        if (Ray(x + 0.5, y + 0.5).z > -0.05) continue;
        e += std::hypot(f.u.row(y / scale)[x / scale] * scale - tu, f.v.row(y / scale)[x / scale] * scale - tv); ++n;
    }
    BlurParams bp; bp.shutter = 0.5f; bp.maxSamples = 64; bp.mode = (m == OLD) ? kBlurObjects : kBlurNatural;
    Res r; ApplyMotionBlur(cur, &f, &b, scale, bp, r.blurred); r.epe = e / n;
    // echte Belichtung: 48 Zwischenbilder ueber 180 Grad Verschluss
    RGBAImage ref; ref.resize(W, H); const int N = 48;
    for (int i = 0; i < N; ++i) { RGBAImage s; Render(camY + speed * (-0.25 + 0.5 * (i + 0.5) / N), s); for (size_t k = 0; k < ref.d.size(); ++k) ref.d[k] += s.d[k] / N; }
    double be = 0; for (size_t k = 0; k < ref.d.size(); k += 4) for (int c = 0; c < 3; ++c) be += std::fabs(r.blurred.d[k + c] - ref.d[k + c]);
    r.blurErr = be / (ref.d.size() / 4 * 3);
    return r;
}

int main(int argc, char** argv)
{
    std::string out = argc > 1 ? argv[1] : ".";
    for (double speed : { 0.6, 2.5 }) {
        double tu, tv; TrueFlow(W / 2.0, H - 5.0, speed, tu, tv);
        std::printf("\nFluggeschwindigkeit %.1f -> Bewegung unten im Bild %.0f px/Frame%s\n", speed, std::hypot(tu, tv), speed > 1 ? " (Speed Ramp)" : "");
        const char* names[] = { "alt (1.3 Objekte)", "Frei + Auffuellen", "Kamera-Modell" };
        for (int m = 0; m < 3; ++m) {
            double sumE = 0, sumB = 0, minB = 1e9, maxB = 0;
            Res last;
            for (int fr = 0; fr < 4; ++fr) {   // 4 aufeinanderfolgende Frames -> Flackern
                Res r = Run(fr * speed, speed, (Method)m);
                sumE += r.epe; sumB += r.blurErr; minB = std::min(minB, r.blurErr); maxB = std::max(maxB, r.blurErr);
                if (fr == 0) last = r;
            }
            std::printf("  %-20s Tracking-Fehler %6.2f px   Abw. zur echten Belichtung %.4f   Schwankung %.4f\n",
                        names[m], sumE / 4, sumB / 4, maxB - minB);
            WritePPM(out + "/drone_s" + std::to_string((int)(speed * 10)) + "_m" + std::to_string(m) + ".ppm", last.blurred);
        }
        RGBAImage ref; ref.resize(W, H); const int N = 48;
        for (int i = 0; i < N; ++i) { RGBAImage s; Render(speed * (-0.25 + 0.5 * (i + 0.5) / N), s); for (size_t k = 0; k < ref.d.size(); ++k) ref.d[k] += s.d[k] / N; }
        WritePPM(out + "/drone_s" + std::to_string((int)(speed * 10)) + "_ref.ppm", ref);
    }
    RGBAImage o; Render(0, o); WritePPM(out + "/drone_orig.ppm", o);
    return 0;
}
