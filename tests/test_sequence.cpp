// test_sequence.cpp - Drohnenflug ueber mehrere Bilder mit Speed Ramp (Beschleunigung).
// Vergleicht die Ausbaustufen der Analyse und Unschaerfe gegen die echte Belichtung:
//   A  Basis (Startwert vom letzten Bild, gerade Bahn)
//   B  + Rueckwaertsbewegung vom vorherigen Bild wiederverwenden
// Gemessen: Tracking-Fehler, Abweichung zur echten Belichtung, Schwankung, Zeit.
#include "../src/core/FlowCore.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace fmb;

static const int W = 360, H = 640;
static const double F = 380, PITCH = 0.42, HEIGHT = 40.0;
struct V3 { double x, y, z; };
static V3 Ray(double px, double py)
{
    double xr = (px - W / 2.0) / F, yr = (py - H / 2.0) / F;
    double c = std::cos(PITCH), s = std::sin(PITCH);
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
// Kamera dreht sich leicht (Gieren) und fliegt vorwaerts: Pos/Gierwinkel je Zeit
static void Render(double camY, double yaw, RGBAImage& img)
{
    img.resize(W, H);
    const double cy = std::cos(yaw), sy = std::sin(yaw);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        float* p = img.px(x, y); p[3] = 1;
        V3 d0 = Ray(x + 0.5, y + 0.5);
        V3 d = { d0.x * cy - d0.y * sy, d0.x * sy + d0.y * cy, d0.z };
        if (d.z > -0.02) { double az = std::atan2(d.x, d.y), el = d.z; float c = (float)(0.6 + 0.2 * std::sin(az * 9) * std::cos(el * 20) + 0.1 * std::sin(az * 23 + el * 11));
            p[0] = c * 0.85f; p[1] = c * 0.9f; p[2] = c; continue; }
        double t = HEIGHT / -d.z; Ground(t * d.x, camY + t * d.y, p);
    }
}
// Projektion eines Bodenpunkts (X,Y) in die Kamera (Position camY, Gierwinkel yaw)
static bool Project(double X, double Y, double camY, double yaw, double& px, double& py)
{
    double gx = X, gy = Y - camY, gz = -HEIGHT;
    double cy = std::cos(-yaw), sy = std::sin(-yaw);
    double rx = gx * cy - gy * sy, ry = gx * sy + gy * cy;
    double c = std::cos(PITCH), s = std::sin(PITCH);
    double Yc = ry * -s + gz * -c, Zc = ry * c + gz * -s;
    if (Zc <= 0.1) return false;
    px = W / 2.0 + F * rx / Zc; py = H / 2.0 + F * Yc / Zc;
    return true;
}
// Echte Bewegung eines Pixels von Zeitpunkt a nach b (false = Himmel)
static double CamY(double t) { return 0.8 * t + 0.35 * t * t; }   // beschleunigt (Speed Ramp)
static double Yaw(double t) { return 0.012 * t; }                 // leichte Drehung
static bool TrueFlow(double px, double py, double ta, double tb, double& u, double& v)
{
    const double yaw = Yaw(ta);
    V3 d0 = Ray(px, py);
    V3 d = { d0.x * std::cos(yaw) - d0.y * std::sin(yaw), d0.x * std::sin(yaw) + d0.y * std::cos(yaw), d0.z };
    if (d.z > -0.02) { u = v = 0; return false; }
    double t = HEIGHT / -d.z; double X = t * d.x, Y = CamY(ta) + t * d.y;
    double qx, qy; if (!Project(X, Y, CamY(tb), Yaw(tb), qx, qy)) { u = v = 0; return false; }
    u = qx - px; v = qy - py; return true;
}

int main(int argc, char** argv)
{
    const double SH = argc > 1 ? std::atof(argv[1]) : 0.5; // Verschluss (Blur Amount)
    const int N = 9, scale = 2;
    std::vector<RGBAImage> frames(N + 1);
    std::vector<Image> luma(N + 1);
    // optional: Sensorrauschen (2. Argument, z. B. 0.03)
    const double NOISE = argc > 2 ? std::atof(argv[2]) : 0.0;
    for (int n = 0; n <= N; ++n) {
        Render(CamY(n), Yaw(n), frames[n]);
        if (NOISE > 0) { unsigned st = 12345u + n * 7919u;
            for (size_t k = 0; k < frames[n].d.size(); ++k) { if ((k & 3) == 3) continue; st = st * 1664525u + 1013904223u;
                frames[n].d[k] += (float)(NOISE * (((st >> 8) & 0xFFFF) / 65535.0 - 0.5) * 2.0); } }
        LumaFromRGBA(frames[n], scale, luma[n]);
    }
    // echte Belichtung (180 Grad) je Bild
    std::vector<RGBAImage> refs(N + 1);
    for (int n = 1; n < N; ++n) {
        RGBAImage& r = refs[n]; r.resize(W, H); const int S = 48;
        for (int i = 0; i < S; ++i) { RGBAImage s; double t = n + SH * (-0.5 + (i + 0.5) / S); Render(CamY(t), Yaw(t), s);
            for (size_t k = 0; k < r.d.size(); ++k) r.d[k] += s.d[k] / S; }
    }
    std::printf("Verschluss %.2f  Speed Ramp: Bewegung", SH); std::printf("  unten im Bild von %.0f auf %.0f px/Bild\n",
                [&] { double u, v; TrueFlow(W / 2.0, H - 5.0, 1, 2, u, v); return std::hypot(u, v); }(),
                [&] { double u, v; TrueFlow(W / 2.0, H - 5.0, N - 1, N, u, v); return std::hypot(u, v); }());

    const char* names[] = { "A Basis", "B + Wiederverwenden" };
    for (int m = 0; m < 2; ++m) {
        const bool reuse = m >= 1;
        CameraPrior prior; bool havePrior = false;
        Flow prevFwd; bool havePrevFwd = false;
        double sumE = 0, sumB = 0, minB = 1e9, maxB = 0, ms = 0; int cnt = 0, reused = 0;
        double sumFl = 0; RGBAImage prevOut; // Flackern: Aenderung des Ergebnisses gegenueber der echten Aenderung
        for (int n = 1; n < N; ++n) {
            FlowParams fp; Flow f, b; CameraInfo ci;
            Flow hint; const Flow* hintP = nullptr;
            if (reuse && havePrevFwd && InvertFlow(prevFwd, hint) > 0.5f) hintP = &hint;
            auto t0 = std::chrono::steady_clock::now();
            CameraFlows(luma[n], &luma[n + 1], &luma[n - 1], fp,
                        [](const Image& a, const Image& b2, const FlowParams& q, Flow& o) { ComputeFlow(a, b2, q, o); },
                        &f, &b, &ci, havePrior ? &prior : nullptr, hintP);
            const Flow rawF = f;
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            reused += ci.reusedB;
            prior.validF = ci.modelF; prior.validB = ci.modelF; // Rueckwaerts = Umkehrung von vorwaerts (exakt)
            std::copy(ci.Hf, ci.Hf + 9, prior.Hf);
            prior.validB = InvertHomography(ci.Hf, prior.Hb) && ci.modelF;
            havePrior = ci.modelF;
            prevFwd = rawF; havePrevFwd = true;
            // Fehler
            double e = 0; int k = 0;
            for (int y = 10; y < H - 10; y += 3) for (int x = 10; x < W - 10; x += 3) {
                double tu, tv, su, sv;
                if (!TrueFlow(x + 0.5, y + 0.5, n, n + 1, tu, tv) || !TrueFlow(x + 0.5, y + 0.5, n, n - 1, su, sv)) continue;
                e += std::hypot(f.u.row(y / scale)[x / scale] * scale - tu, f.v.row(y / scale)[x / scale] * scale - tv);
                e += std::hypot(b.u.row(y / scale)[x / scale] * scale - su, b.v.row(y / scale)[x / scale] * scale - sv);
                k += 2;
            }
            BlurParams bp; bp.shutter = (float)SH; bp.maxSamples = 64;
            RGBAImage out; ApplyMotionBlur(frames[n], &f, &b, scale, bp, out);
            double be = 0; for (size_t q = 0; q < out.d.size(); q += 4) for (int c = 0; c < 3; ++c) be += std::fabs(out.d[q + c] - refs[n].d[q + c]);
            be /= (out.d.size() / 4 * 3);
            // Flackern: |(out_n - out_{n-1}) - (ref_n - ref_{n-1})|
            if (n > 1) { double fl = 0; for (size_t q = 0; q < out.d.size(); q += 4) for (int c = 0; c < 3; ++c)
                fl += std::fabs((out.d[q + c] - prevOut.d[q + c]) - (refs[n].d[q + c] - refs[n - 1].d[q + c]));
                sumFl += fl / (out.d.size() / 4 * 3); }
            prevOut = out;
            sumE += e / k; sumB += be; minB = std::min(minB, be); maxB = std::max(maxB, be); ++cnt;
        }
        std::printf("  %-22s Fehler %5.2f px   Abw. Belichtung %.5f   Flackern %.5f   Analyse %4.0f ms/Bild   wiederverwendet %d/%d\n",
                    names[m], sumE / cnt, sumB / cnt, sumFl / (cnt - 1), ms / cnt, reused, cnt);
    }
    return 0;
}
