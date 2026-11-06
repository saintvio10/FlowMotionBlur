// FlowCore.cpp - Implementierung des host-unabhaengigen Kerns.
//
// Hinweis: src/gpu/Kernels.h enthaelt dieselben Schritte als OpenCL-Kernel.
// Aenderungen am Algorithmus muessen an beiden Stellen erfolgen
// (tests/test_core.cpp vergleicht beide Ergebnisse).

#include "FlowCore.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>

namespace fmb {

// ---------------------------------------------------------------------------
// Hilfsfunktionen
// ---------------------------------------------------------------------------

void ParallelFor(int begin, int end, const std::function<void(int, int)>& fn)
{
    const int range = end - begin;
    if (range <= 0) return;

    unsigned hw = std::thread::hardware_concurrency();
    int threads = static_cast<int>(hw == 0 ? 4 : hw);
    threads = std::min(threads, range);
    if (threads <= 1 || range < 16) { fn(begin, end); return; }

    const int grain = std::max(1, range / (threads * 4));
    std::atomic<int> next(begin);
    auto worker = [&]() {
        for (;;) {
            int b = next.fetch_add(grain);
            if (b >= end) break;
            fn(b, std::min(end, b + grain));
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (int i = 0; i < threads - 1; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
}

static inline float Clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Bilineares Abtasten mit Randbegrenzung.
static inline float Sample(const Image& im, float x, float y)
{
    x = Clampf(x, 0.f, static_cast<float>(im.w - 1));
    y = Clampf(y, 0.f, static_cast<float>(im.h - 1));
    int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, im.w - 1), y1 = std::min(y0 + 1, im.h - 1);
    float fx = x - x0, fy = y - y0;
    const float* r0 = im.row(y0);
    const float* r1 = im.row(y1);
    float a = r0[x0] + (r0[x1] - r0[x0]) * fx;
    float b = r1[x0] + (r1[x1] - r1[x0]) * fx;
    return a + (b - a) * fy;
}

static inline void SampleRGBA(const RGBAImage& im, float x, float y, float out[4])
{
    x = Clampf(x, 0.f, static_cast<float>(im.w - 1));
    y = Clampf(y, 0.f, static_cast<float>(im.h - 1));
    int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, im.w - 1), y1 = std::min(y0 + 1, im.h - 1);
    float fx = x - x0, fy = y - y0;
    const float* p00 = im.px(x0, y0);
    const float* p10 = im.px(x1, y0);
    const float* p01 = im.px(x0, y1);
    const float* p11 = im.px(x1, y1);
    float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    for (int c = 0; c < 4; ++c)
        out[c] = p00[c] * w00 + p10[c] * w10 + p01[c] * w01 + p11[c] * w11;
}

// Separabler Mittelwertfilter (Radius r), Raender werden korrekt normalisiert.
static void BoxFilter(const Image& src, int r, Image& dst)
{
    const int w = src.w, h = src.h;
    if (r <= 0) { dst = src; return; }

    Image tmp; tmp.resize(w, h);

    // horizontal
    ParallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const float* s = src.row(y);
            float* t = tmp.row(y);
            double sum = 0;
            for (int x = 0; x <= std::min(r, w - 1); ++x) sum += s[x];
            for (int x = 0; x < w; ++x) {
                int a = std::max(0, x - r), b = std::min(w - 1, x + r);
                t[x] = static_cast<float>(sum / (b - a + 1));
                if (x - r >= 0) sum -= s[x - r];
                if (x + r + 1 < w) sum += s[x + r + 1];
            }
        }
    });

    // vertikal (spaltenblockweise, zeilenweiser Speicherzugriff)
    dst.resize(w, h);
    ParallelFor(0, w, [&](int x0, int x1) {
        const int n = x1 - x0;
        std::vector<double> sum(n, 0.0);
        for (int y = 0; y <= std::min(r, h - 1); ++y) {
            const float* t = tmp.row(y) + x0;
            for (int i = 0; i < n; ++i) sum[i] += t[i];
        }
        for (int y = 0; y < h; ++y) {
            int a = std::max(0, y - r), b = std::min(h - 1, y + r);
            double inv = 1.0 / (b - a + 1);
            float* d = dst.row(y) + x0;
            for (int i = 0; i < n; ++i) d[i] = static_cast<float>(sum[i] * inv);
            if (y - r >= 0) {
                const float* t = tmp.row(y - r) + x0;
                for (int i = 0; i < n; ++i) sum[i] -= t[i];
            }
            if (y + r + 1 < h) {
                const float* t = tmp.row(y + r + 1) + x0;
                for (int i = 0; i < n; ++i) sum[i] += t[i];
            }
        }
    });
}

// 3x3-Median, entfernt Ausreisser im Vektorfeld.
static void Median3(const Image& src, Image& dst)
{
    const int w = src.w, h = src.h;
    dst.resize(w, h);
    ParallelFor(0, h, [&](int y0, int y1) {
        float v[9];
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < w; ++x) {
                int k = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const float* r = src.row(std::min(std::max(y + dy, 0), h - 1));
                    for (int dx = -1; dx <= 1; ++dx)
                        v[k++] = r[std::min(std::max(x + dx, 0), w - 1)];
                }
                std::nth_element(v, v + 4, v + 9);
                dst.row(y)[x] = v[4];
            }
        }
    });
}

void Downscale(const Image& src, int factor, Image& dst)
{
    if (factor <= 1) { dst = src; return; }
    const int w = (src.w + factor - 1) / factor;
    const int h = (src.h + factor - 1) / factor;
    dst.resize(w, h);
    ParallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            int sy0 = y * factor, sy1 = std::min(src.h, sy0 + factor);
            for (int x = 0; x < w; ++x) {
                int sx0 = x * factor, sx1 = std::min(src.w, sx0 + factor);
                float s = 0; int n = 0;
                for (int sy = sy0; sy < sy1; ++sy) {
                    const float* r = src.row(sy);
                    for (int sx = sx0; sx < sx1; ++sx) { s += r[sx]; ++n; }
                }
                dst.row(y)[x] = s / n;
            }
        }
    });
}

void LumaFromRGBA(const RGBAImage& src, int factor, Image& dst, float alphaWeight)
{
    factor = std::max(1, factor);
    const int w = (src.w + factor - 1) / factor;
    const int h = (src.h + factor - 1) / factor;
    dst.resize(w, h);
    ParallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            int sy0 = y * factor, sy1 = std::min(src.h, sy0 + factor);
            for (int x = 0; x < w; ++x) {
                int sx0 = x * factor, sx1 = std::min(src.w, sx0 + factor);
                float s = 0; int n = 0;
                for (int sy = sy0; sy < sy1; ++sy) {
                    for (int sx = sx0; sx < sx1; ++sx) {
                        const float* p = src.px(sx, sy);
                        float l = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
                        s += l * (1.f - alphaWeight) + p[3] * alphaWeight;
                        ++n;
                    }
                }
                dst.row(y)[x] = s / n;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Track Points
// ---------------------------------------------------------------------------
// Jeder Punkt wirkt mit einer Gauss-Glocke. Nahe am Punkt wird der Vektor
// (fast) exakt auf die vorgegebene Bewegung gesetzt, weiter weg bleibt der
// berechnete Fluss erhalten.

void ApplyGuides(Flow& f, const std::vector<Guide>& guides, float radius, float scale)
{
    if (guides.empty() || f.empty()) return;
    const float sigma = std::max(1.f, radius * scale);
    const float k = -1.f / (2.f * sigma * sigma);
    const int w = f.w(), h = f.h();
    ParallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                float S = 0, gu = 0, gv = 0;
                for (const Guide& g : guides) {
                    float dx = x - g.x * scale, dy = y - g.y * scale;
                    float wgt = std::exp((dx * dx + dy * dy) * k);
                    S += wgt; gu += wgt * g.dx * scale; gv += wgt * g.dy * scale;
                }
                if (S < 1e-4f) continue;
                float n = std::max(1.f, S), keep = 1.f - std::min(1.f, S);
                f.u.row(y)[x] = f.u.row(y)[x] * keep + gu / n;
                f.v.row(y)[x] = f.v.row(y)[x] * keep + gv / n;
            }
    });
}

// ---------------------------------------------------------------------------
// Optischer Fluss: pyramidales Lucas-Kanade mit Warping
// ---------------------------------------------------------------------------

void ComputeFlow(const Image& I0, const Image& I1, const FlowParams& p, Flow& out,
                 const std::vector<Guide>* guides)
{
    // Pyramiden aufbauen
    std::vector<Image> P0(1, I0), P1(1, I1);
    while (static_cast<int>(P0.size()) < std::max(1, p.maxLevels)) {
        const Image& last = P0.back();
        if (std::min(last.w, last.h) / 2 < 16) break;
        Image a, b;
        Downscale(P0.back(), 2, a);
        Downscale(P1.back(), 2, b);
        P0.push_back(std::move(a));
        P1.push_back(std::move(b));
    }

    const int levels = static_cast<int>(P0.size());
    Flow cur;

    for (int L = levels - 1; L >= 0; --L) {
        const Image& A = P0[L];
        const Image& B = P1[L];
        const int w = A.w, h = A.h;

        // Startwert: hochskalierter Fluss der groeberen Stufe
        Flow init;
        init.u.resize(w, h);
        init.v.resize(w, h);
        if (!cur.empty()) {
            const float sx = static_cast<float>(cur.w()) / w;
            const float sy = static_cast<float>(cur.h()) / h;
            const float ku = 1.f / sx, kv = 1.f / sy;
            ParallelFor(0, h, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < w; ++x) {
                        float cx = (x + 0.5f) * sx - 0.5f;
                        float cy = (y + 0.5f) * sy - 0.5f;
                        init.u.row(y)[x] = Sample(cur.u, cx, cy) * ku;
                        init.v.row(y)[x] = Sample(cur.v, cx, cy) * kv;
                    }
            });
        } else {
            // Groebste Stufe: Startfeld bzw. globale Verschiebung
            CoarseInit(p, I0.w, w, h, init);
            // Track Points geben zusaetzlich lokale Startwerte
            if (guides) ApplyGuides(init, *guides, p.guideRadius, static_cast<float>(w) / I0.w);
        }
        cur = std::move(init);

        // Gradienten von I0 und Strukturtensor (konstant pro Stufe)
        Image Ix, Iy, Ixx, Ixy, Iyy;
        Ix.resize(w, h); Iy.resize(w, h);
        Ixx.resize(w, h); Ixy.resize(w, h); Iyy.resize(w, h);
        ParallelFor(0, h, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                const float* rm = A.row(std::max(y - 1, 0));
                const float* r  = A.row(y);
                const float* rp = A.row(std::min(y + 1, h - 1));
                for (int x = 0; x < w; ++x) {
                    float gx = 0.5f * (r[std::min(x + 1, w - 1)] - r[std::max(x - 1, 0)]);
                    float gy = 0.5f * (rp[x] - rm[x]);
                    Ix.row(y)[x] = gx; Iy.row(y)[x] = gy;
                    Ixx.row(y)[x] = gx * gx; Ixy.row(y)[x] = gx * gy; Iyy.row(y)[x] = gy * gy;
                }
            }
        });
        Image A11, A12, A22;
        BoxFilter(Ixx, p.windowRadius, A11);
        BoxFilter(Ixy, p.windowRadius, A12);
        BoxFilter(Iyy, p.windowRadius, A22);

        Image Ixt, Iyt, B1, B2;
        Ixt.resize(w, h); Iyt.resize(w, h);

        for (int it = 0; it < std::max(1, p.iterations); ++it) {
            // I1 mit aktuellem Fluss warpen, zeitliche Ableitung bilden
            ParallelFor(0, h, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < w; ++x) {
                        const float tx = x + cur.u.row(y)[x], ty = y + cur.v.row(y)[x];
                        // Ziel ausserhalb des Bildes: keine verlaessliche Information
                        // (neuer Bildinhalt) -> keine Aktualisierung, Nachbarn bestimmen den Wert.
                        const bool inside = tx >= 0.f && ty >= 0.f && tx <= w - 1.f && ty <= h - 1.f;
                        float It = inside ? Sample(B, tx, ty) - A.row(y)[x] : 0.f;
                        Ixt.row(y)[x] = Ix.row(y)[x] * It;
                        Iyt.row(y)[x] = Iy.row(y)[x] * It;
                    }
            });
            BoxFilter(Ixt, p.windowRadius, B1);
            BoxFilter(Iyt, p.windowRadius, B2);

            // 2x2-System pro Pixel loesen
            const float reg = p.regularization;
            ParallelFor(0, h, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < w; ++x) {
                        float a = A11.row(y)[x] + reg;
                        float c = A22.row(y)[x] + reg;
                        float b = A12.row(y)[x];
                        float det = a * c - b * b;
                        if (det <= 1e-12f) continue;
                        float b1 = B1.row(y)[x], b2 = B2.row(y)[x];
                        float du = -(c * b1 - b * b2) / det;
                        float dv = -(a * b2 - b * b1) / det;
                        cur.u.row(y)[x] += Clampf(du, -2.f, 2.f);
                        cur.v.row(y)[x] += Clampf(dv, -2.f, 2.f);
                    }
            });
        }

        // Ausreisser entfernen
        Image mu, mv;
        Median3(cur.u, mu);
        Median3(cur.v, mv);
        cur.u = std::move(mu);
        cur.v = std::move(mv);
    }

    if (p.smoothRadius > 0) {
        BoxFilter(cur.u, p.smoothRadius, out.u);
        BoxFilter(cur.v, p.smoothRadius, out.v);
    } else {
        out = std::move(cur);
    }

    if (guides) ApplyGuides(out, *guides, p.guideRadius, 1.f);
}

// ---------------------------------------------------------------------------
// Verlaesslichkeit + Auffuellen unsicherer Vektoren
// ---------------------------------------------------------------------------

// Verlaesslichkeit je Vektor (0..1): Bildstruktur und, falls beide Richtungen
// vorliegen, deren Uebereinstimmung (gleichmaessige Bewegung: vor ~ -rueck).
static void Confidence(const Image& I0, const Flow* fwd, const Flow* bwd, Image& C)
{
    const int w = I0.w, h = I0.h;
    Image Ixx, Ixy, Iyy, Sxx, Sxy, Syy;
    Ixx.resize(w, h); Ixy.resize(w, h); Iyy.resize(w, h);
    ParallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const float* rm = I0.row(std::max(y - 1, 0));
            const float* r  = I0.row(y);
            const float* rp = I0.row(std::min(y + 1, h - 1));
            for (int x = 0; x < w; ++x) {
                float gx = 0.5f * (r[std::min(x + 1, w - 1)] - r[std::max(x - 1, 0)]);
                float gy = 0.5f * (rp[x] - rm[x]);
                Ixx.row(y)[x] = gx * gx; Ixy.row(y)[x] = gx * gy; Iyy.row(y)[x] = gy * gy;
            }
        }
    });
    BoxFilter(Ixx, 3, Sxx); BoxFilter(Ixy, 3, Sxy); BoxFilter(Iyy, 3, Syy);

    const float tau = 2e-5f;
    C.resize(w, h);
    ParallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                float a = Sxx.row(y)[x], b = Sxy.row(y)[x], c = Syy.row(y)[x];
                float lmin = std::max(0.f, 0.5f * (a + c - std::sqrt((a - c) * (a - c) + 4.f * b * b)));
                float conf = lmin / (lmin + tau);
                if (fwd && bwd) {
                    float fu = fwd->u.row(y)[x], fv = fwd->v.row(y)[x];
                    float bu = bwd->u.row(y)[x], bv = bwd->v.row(y)[x];
                    float e = std::hypot(fu + bu, fv + bv);
                    float m = 0.5f * (std::hypot(fu, fv) + std::hypot(bu, bv));
                    float sigma = 0.5f + 0.25f * m;
                    conf *= std::exp(-(e * e) / (2.f * sigma * sigma));
                }
                C.row(y)[x] = conf;
            }
    });
}

// Weicher Filter (zweimal Box = Dreiecksprofil), vermeidet eckige Bloecke.
static void SoftFilter(const Image& src, int r, Image& dst)
{
    Image t;
    BoxFilter(src, std::max(1, r / 2), t);
    BoxFilter(t, std::max(1, r - r / 2), dst);
}

void RefineFlows(const Image& I0, Flow* fwd, Flow* bwd)
{
    if (fwd && fwd->empty()) fwd = nullptr;
    if (bwd && bwd->empty()) bwd = nullptr;
    if (!fwd && !bwd) return;
    const int w = I0.w, h = I0.h;
    if ((fwd && (fwd->w() != w || fwd->h() != h)) || (bwd && (bwd->w() != w || bwd->h() != h))) return;

    Image C;
    Confidence(I0, fwd, bwd, C);

    // Normalisierte Faltung mit mehreren Radien: der kleinste Radius, in dem
    // genug verlaessliche Vektoren liegen, liefert den Ersatzwert.
    const int radii[] = { 4, 12, 32 };
    const int nR = 3;
    std::vector<Image> BC(nR);
    for (int i = 0; i < nR; ++i) SoftFilter(C, radii[i], BC[i]);

    auto refine = [&](Flow& f) {
        Image cu, cv; cu.resize(w, h); cv.resize(w, h);
        for (size_t i = 0; i < C.d.size(); ++i) { cu.d[i] = C.d[i] * f.u.d[i]; cv.d[i] = C.d[i] * f.v.d[i]; }
        std::vector<Image> BU(nR), BV(nR);
        for (int i = 0; i < nR; ++i) { SoftFilter(cu, radii[i], BU[i]); SoftFilter(cv, radii[i], BV[i]); }
        ParallelFor(0, h, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < w; ++x) {
                    // weich zwischen den Radien ueberblenden statt hart umzuschalten
                    float accU = 0, accV = 0, remain = 1.f;
                    for (int i = 0; i < nR && remain > 1e-3f; ++i) {
                        float bc = BC[i].row(y)[x];
                        if (bc <= 1e-5f) continue;
                        float wgt = (i == nR - 1) ? remain : remain * std::min(1.f, bc / 0.15f);
                        accU += wgt * BU[i].row(y)[x] / bc;
                        accV += wgt * BV[i].row(y)[x] / bc;
                        remain -= wgt;
                    }
                    if (remain > 0.999f) continue; // nirgends verlaessliche Werte
                    float fu = accU / (1.f - remain), fv = accV / (1.f - remain);
                    float c = C.row(y)[x];
                    f.u.row(y)[x] = c * f.u.row(y)[x] + (1.f - c) * fu;
                    f.v.row(y)[x] = c * f.v.row(y)[x] + (1.f - c) * fv;
                }
        });
    };
    if (fwd) refine(*fwd);
    if (bwd) refine(*bwd);
}

// ---------------------------------------------------------------------------
// Globale Vorsuche
// ---------------------------------------------------------------------------

void EstimateGlobalShift(const Image& I0, const Image& I1, float& dx, float& dy)
{
    dx = dy = 0.f;
    if (I0.empty() || I0.w != I1.w || I0.h != I1.h) return;
    // auf max. ~96 px Kantenlaenge verkleinern
    const int f = std::max(1, (std::max(I0.w, I0.h) + 95) / 96);
    Image a, b;
    Downscale(I0, f, a);
    Downscale(I1, f, b);
    const int w = a.w, h = a.h;
    const int R = std::max(2, std::min(w, h) / 3); // Suchbereich +- R (verkleinerte px)
    // Mittelwert entfernen (robuster bei Helligkeitsaenderung)
    auto meanOf = [](const Image& im) { double s = 0; for (float v : im.d) s += v; return static_cast<float>(s / std::max<size_t>(1, im.d.size())); };
    const float ma = meanOf(a), mb = meanOf(b);
    double best = 1e30; int bx = 0, by = 0;
    for (int sy = -R; sy <= R; ++sy)
        for (int sx = -R; sx <= R; ++sx) {
            // Ueberlappung muss gross genug sein
            int x0 = std::max(0, -sx), x1 = std::min(w, w - sx);
            int y0 = std::max(0, -sy), y1 = std::min(h, h - sy);
            int ow = x1 - x0, oh = y1 - y0;
            if (ow < w / 2 || oh < h / 2) continue;
            double sad = 0;
            for (int y = y0; y < y1; ++y) {
                const float* ra = a.row(y);
                const float* rb = b.row(y + sy);
                for (int x = x0; x < x1; ++x) sad += std::fabs((ra[x] - ma) - (rb[x + sx] - mb));
            }
            sad /= (ow * oh);
            // leichte Bevorzugung kleiner Verschiebungen (gegen Mehrdeutigkeit)
            sad *= 1.0 + 0.002 * (sx * sx + sy * sy);
            if (sad < best) { best = sad; bx = sx; by = sy; }
        }
    dx = static_cast<float>(bx * f);
    dy = static_cast<float>(by * f);
}

// ---------------------------------------------------------------------------
// Kameramodell (Homographie, RANSAC)
// ---------------------------------------------------------------------------

// Loest A x = b (n x n, Gauss mit Pivotsuche). false bei Singularitaet.
static bool SolveLinear(std::vector<double>& A, std::vector<double>& b, int n)
{
    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int r = c + 1; r < n; ++r) if (std::fabs(A[r * n + c]) > std::fabs(A[piv * n + c])) piv = r;
        if (std::fabs(A[piv * n + c]) < 1e-12) return false;
        if (piv != c) { for (int k = 0; k < n; ++k) std::swap(A[c * n + k], A[piv * n + k]); std::swap(b[c], b[piv]); }
        for (int r = c + 1; r < n; ++r) {
            double f = A[r * n + c] / A[c * n + c];
            for (int k = c; k < n; ++k) A[r * n + k] -= f * A[c * n + k];
            b[r] -= f * b[c];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        double s = b[r];
        for (int k = r + 1; k < n; ++k) s -= A[r * n + k] * b[k];
        b[r] = s / A[r * n + r];
    }
    return true;
}

struct Corr { double x, y, u, v; };

// Homographie (h33 = 1) per kleinster Quadrate aus normierten Punkten.
static bool FitH(const std::vector<Corr>& c, const std::vector<int>& idx, double H[9])
{
    if (idx.size() < 4) return false;
    std::vector<double> AtA(64, 0.0), Atb(8, 0.0);
    for (int i : idx) {
        const Corr& p = c[i];
        double r1[8] = { p.x, p.y, 1, 0, 0, 0, -p.u * p.x, -p.u * p.y };
        double r2[8] = { 0, 0, 0, p.x, p.y, 1, -p.v * p.x, -p.v * p.y };
        for (int a = 0; a < 8; ++a) {
            Atb[a] += r1[a] * p.u + r2[a] * p.v;
            for (int b = 0; b < 8; ++b) AtA[a * 8 + b] += r1[a] * r1[b] + r2[a] * r2[b];
        }
    }
    if (!SolveLinear(AtA, Atb, 8)) return false;
    for (int i = 0; i < 8; ++i) H[i] = Atb[i];
    H[8] = 1.0;
    return true;
}

static inline void ApplyH(const double H[9], double x, double y, double& u, double& v)
{
    double z = H[6] * x + H[7] * y + H[8];
    if (std::fabs(z) < 1e-9) z = 1e-9;
    u = (H[0] * x + H[1] * y + H[2]) / z;
    v = (H[3] * x + H[4] * y + H[5]) / z;
}

// Schaetzt die Homographie fuer ein Vektorfeld und ersetzt es durch das Modell.
static bool FitOne(Flow& f, const Image& C)
{
    const int w = f.w(), h = f.h();
    // Stichproben auf einem Raster, nur verlaessliche Vektoren
    const int step = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(w) * h / 3000.0)));
    // Normierung: Mittelpunkt 0, Ausdehnung ~1
    const double cx = 0.5 * w, cy = 0.5 * h, sc = 2.0 / std::max(w, h);
    std::vector<Corr> c;
    std::vector<double> lens;
    for (int y = step / 2; y < h; y += step)
        for (int x = step / 2; x < w; x += step) {
            if (C.row(y)[x] < 0.3f) continue;
            double fu = f.u.row(y)[x], fv = f.v.row(y)[x];
            c.push_back({ (x - cx) * sc, (y - cy) * sc, (x + fu - cx) * sc, (y + fv - cy) * sc });
            lens.push_back(std::hypot(fu, fv));
        }
    if (c.size() < 40) return false;
    std::nth_element(lens.begin(), lens.begin() + lens.size() / 2, lens.end());
    const double medLen = lens[lens.size() / 2];
    const double thr = std::max(0.75, 0.05 * medLen) * sc; // Toleranz (normiert)

    auto inliersOf = [&](const double H[9], std::vector<int>& out) {
        out.clear();
        for (int i = 0; i < static_cast<int>(c.size()); ++i) {
            double u, v; ApplyH(H, c[i].x, c[i].y, u, v);
            if (std::hypot(u - c[i].u, v - c[i].v) < thr) out.push_back(i);
        }
    };

    // RANSAC mit festem Startwert -> gleiche Eingabe, gleiches Ergebnis (kein Flackern)
    uint32_t rng = 12345u;
    auto rnd = [&](int n) { rng = rng * 1664525u + 1013904223u; return static_cast<int>((rng >> 8) % static_cast<uint32_t>(n)); };
    double bestH[9]; std::vector<int> bestIn, in;
    const int n = static_cast<int>(c.size());
    for (int it = 0; it < 500; ++it) {
        std::vector<int> pick;
        while (pick.size() < 4) { int k = rnd(n); if (std::find(pick.begin(), pick.end(), k) == pick.end()) pick.push_back(k); }
        double H[9];
        if (!FitH(c, pick, H)) continue;
        inliersOf(H, in);
        if (in.size() > bestIn.size()) { bestIn = in; std::copy(H, H + 9, bestH); }
    }
    if (bestIn.size() < std::max<size_t>(30, c.size() / 4)) return false;
    // Nachoptimieren auf allen Inliern
    for (int r = 0; r < 2; ++r) {
        double H[9];
        if (!FitH(c, bestIn, H)) break;
        std::copy(H, H + 9, bestH);
        inliersOf(bestH, bestIn);
    }

    // Modellfeld schreiben
    ParallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                double nx = (x - cx) * sc, ny = (y - cy) * sc, u, v;
                ApplyH(bestH, nx, ny, u, v);
                f.u.row(y)[x] = static_cast<float>(u / sc + cx - x);
                f.v.row(y)[x] = static_cast<float>(v / sc + cy - y);
            }
    });
    return true;
}

bool FitCameraModel(const Image& I0, Flow* fwd, Flow* bwd)
{
    if (fwd && fwd->empty()) fwd = nullptr;
    if (bwd && bwd->empty()) bwd = nullptr;
    if (!fwd && !bwd) return false;
    const int w = I0.w, h = I0.h;
    if ((fwd && (fwd->w() != w || fwd->h() != h)) || (bwd && (bwd->w() != w || bwd->h() != h))) return false;
    Image C;
    Confidence(I0, fwd, bwd, C);
    bool ok = true;
    if (fwd) ok = FitOne(*fwd, C) && ok;
    if (bwd) ok = FitOne(*bwd, C) && ok;
    return ok;
}

// ---------------------------------------------------------------------------
// Bewegungsunschaerfe
// ---------------------------------------------------------------------------
// Rekonstruktionsfilter nach McGuire et al. 2012 ("A Reconstruction Filter for
// Plausible Motion Blur"). Statt Tiefe wird die Geschwindigkeit als Hinweis auf
// den Vordergrund genutzt: Schnelleres liegt vorne und darf ueber Langsameres
// hinaus verschmieren. Dadurch werden auch Kanten bewegter Objekte weich.

static inline float Cone(float dist, float len)     { return len > 1e-4f ? Clampf(1.f - dist / len, 0.f, 1.f) : 0.f; }
static inline float Cylinder(float dist, float len)
{
    float a = 0.95f * len, b = 1.05f * len + 1e-4f;
    float t = Clampf((dist - a) / (b - a), 0.f, 1.f);
    return 1.f - t * t * (3.f - 2.f * t);
}

void ApplyMotionBlur(const RGBAImage& src, const Flow* fwd, const Flow* bwd,
                     int flowScale, const BlurParams& p, RGBAImage& dst)
{
    const int W = src.w, H = src.h;
    dst.resize(W, H);
    if (fwd && fwd->empty()) fwd = nullptr;
    if (bwd && bwd->empty()) bwd = nullptr;
    if ((!fwd && !bwd) || p.shutter <= 0.f || W <= 0 || H <= 0) {
        dst.d = src.d;
        return;
    }

    const float scale = static_cast<float>(std::max(1, flowScale));
    const float inv = 1.f / scale;
    const float half = 0.5f * p.shutter;

    // 1) Halbe Belichtungs-Strecke pro Pixel in voller Aufloesung.
    //    Geschwindigkeit = Mittel aus Vorwaerts- und (negiertem) Rueckwaertsfluss.
    Image VX, VY;
    VX.resize(W, H); VY.resize(W, H);
    std::vector<float> rowMax(H, 0.f);
    ParallelFor(0, H, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            float m = 0.f;
            for (int x = 0; x < W; ++x) {
                float fx = (x + 0.5f) * inv - 0.5f, fy = (y + 0.5f) * inv - 0.5f;
                float vx, vy;
                if (fwd && bwd) {
                    vx = 0.5f * (Sample(fwd->u, fx, fy) - Sample(bwd->u, fx, fy));
                    vy = 0.5f * (Sample(fwd->v, fx, fy) - Sample(bwd->v, fx, fy));
                } else if (fwd) {
                    vx = Sample(fwd->u, fx, fy); vy = Sample(fwd->v, fx, fy);
                } else {
                    vx = -Sample(bwd->u, fx, fy); vy = -Sample(bwd->v, fx, fy);
                }
                vx *= scale * half; vy *= scale * half;
                float len = std::sqrt(vx * vx + vy * vy);
                if (len > kMaxBlurRadius) { vx *= kMaxBlurRadius / len; vy *= kMaxBlurRadius / len; len = kMaxBlurRadius; }
                VX.row(y)[x] = vx; VY.row(y)[x] = vy;
                m = std::max(m, len);
            }
            rowMax[y] = m;
        }
    });
    float globalMax = 0.f;
    for (float m : rowMax) globalMax = std::max(globalMax, m);
    if (globalMax < 0.5f) { dst.d = src.d; return; }

    if (p.mode == kBlurNatural) {
        // Jeder Pixel mittelt das Bild entlang seiner eigenen Bewegungsbahn
        // (gleichmaessige Belichtung von -halbe bis +halbe Verschlusszeit).
        const int maxS = std::max(1, p.maxSamples);
        ParallelFor(0, H, [&](int y0, int y1) {
            float c[4];
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x) {
                    const float vx = VX.row(y)[x], vy = VY.row(y)[x];
                    const float len = std::sqrt(vx * vx + vy * vy);
                    const float* sx = src.px(x, y);
                    float* o = dst.px(x, y);
                    if (len < 0.25f) { o[0] = sx[0]; o[1] = sx[1]; o[2] = sx[2]; o[3] = sx[3]; continue; }
                    int S = std::min(maxS, static_cast<int>(std::ceil(2.f * len)) + 1);
                    if ((S & 1) == 0) ++S;
                    float acc[4] = { 0, 0, 0, 0 };
                    for (int i = 0; i < S; ++i) {
                        float t = (S == 1) ? 0.f : -1.f + 2.f * i / (S - 1);
                        SampleRGBA(src, x + vx * t, y + vy * t, c);
                        for (int k = 0; k < 4; ++k) acc[k] += c[k];
                    }
                    const float inv = 1.f / S;
                    o[0] = acc[0] * inv; o[1] = acc[1] * inv; o[2] = acc[2] * inv; o[3] = acc[3] * inv;
                }
        });
        return;
    }

    // 2) Kachel-Maximum (dominante Bewegung je Kachel) und 3x3-Nachbarschaftsmaximum.
    const int K = std::max(4, static_cast<int>(std::ceil(globalMax)));
    const int TW = (W + K - 1) / K, TH = (H + K - 1) / K;
    std::vector<float> tmx(TW * TH, 0.f), tmy(TW * TH, 0.f);
    ParallelFor(0, TH, [&](int t0, int t1) {
        for (int ty = t0; ty < t1; ++ty)
            for (int tx = 0; tx < TW; ++tx) {
                float bx = 0, by = 0, bl = -1;
                for (int y = ty * K; y < std::min(H, ty * K + K); ++y)
                    for (int x = tx * K; x < std::min(W, tx * K + K); ++x) {
                        float vx = VX.row(y)[x], vy = VY.row(y)[x];
                        float l = vx * vx + vy * vy;
                        if (l > bl) { bl = l; bx = vx; by = vy; }
                    }
                tmx[ty * TW + tx] = bx; tmy[ty * TW + tx] = by;
            }
    });
    std::vector<float> nmx(TW * TH, 0.f), nmy(TW * TH, 0.f);
    for (int ty = 0; ty < TH; ++ty)
        for (int tx = 0; tx < TW; ++tx) {
            float bx = 0, by = 0, bl = -1;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int x = tx + dx, y = ty + dy;
                    if (x < 0 || y < 0 || x >= TW || y >= TH) continue;
                    float vx = tmx[y * TW + x], vy = tmy[y * TW + x];
                    float l = vx * vx + vy * vy;
                    if (l > bl) { bl = l; bx = vx; by = vy; }
                }
            nmx[ty * TW + tx] = bx; nmy[ty * TW + tx] = by;
        }

    // 3) Rekonstruktion.
    const int maxS = std::max(3, p.maxSamples);

    ParallelFor(0, H, [&](int y0, int y1) {
        float c[4];
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < W; ++x) {
                const int ti = (y / K) * TW + (x / K);
                const float nx = nmx[ti], ny = nmy[ti];
                const float nLen = std::sqrt(nx * nx + ny * ny);
                const float* sx = src.px(x, y);
                float* o = dst.px(x, y);
                if (nLen < 0.5f) {
                    o[0] = sx[0]; o[1] = sx[1]; o[2] = sx[2]; o[3] = sx[3];
                    continue;
                }

                const float vxX = VX.row(y)[x], vyX = VY.row(y)[x];
                const float lenX = std::max(0.5f, std::sqrt(vxX * vxX + vyX * vyX));

                float wsum = 1.f / lenX;
                float acc[4] = { sx[0] * wsum, sx[1] * wsum, sx[2] * wsum, sx[3] * wsum };

                int S = std::min(maxS, std::max(3, static_cast<int>(std::ceil(2.f * nLen))));
                if ((S & 1) == 0) ++S; // ungerade -> Mitte wird nicht doppelt getroffen
                // Abwechselnd entlang der dominanten Bewegung der Umgebung (vorne liegende
                // Objekte verschmieren hinein) und der eigenen Bewegung des Pixels
                // abtasten (verhindert falsche Richtung, wenn die Umgebung anders laeuft).
                const bool ownDir = std::sqrt(vxX * vxX + vyX * vyX) >= 0.5f;
                for (int i = 0; i < S; ++i) {
                    float t = -1.f + 2.f * (i + 1.f) / (S + 1.f);
                    const bool useOwn = ownDir && (i & 1);
                    float px = x + (useOwn ? vxX : nx) * t, py = y + (useOwn ? vyX : ny) * t;
                    int ix = static_cast<int>(std::floor(px + 0.5f)), iy = static_cast<int>(std::floor(py + 0.5f));
                    if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;
                    if (ix == x && iy == y) continue;

                    const float vxY = VX.row(iy)[ix], vyY = VY.row(iy)[ix];
                    const float lenY = std::max(0.5f, std::sqrt(vxY * vxY + vyY * vyY));
                    const float dist = std::sqrt((px - x) * (px - x) + (py - y) * (py - y));

                    // "Tiefe" = Geschwindigkeit: f=1 wenn Y vorne, b=1 wenn Y hinten
                    float f = Clampf(1.f - (lenX - lenY) / kSoftZ, 0.f, 1.f);
                    float b = Clampf(1.f - (lenY - lenX) / kSoftZ, 0.f, 1.f);

                    float a = f * Cone(dist, lenY)
                            + b * Cone(dist, lenX)
                            + 2.f * Cylinder(dist, lenY) * Cylinder(dist, lenX);
                    if (a <= 0.f) continue;

                    SampleRGBA(src, px, py, c);
                    wsum += a;
                    for (int k = 0; k < 4; ++k) acc[k] += c[k] * a;
                }
                const float iw = 1.f / wsum;
                o[0] = acc[0] * iw; o[1] = acc[1] * iw; o[2] = acc[2] * iw; o[3] = acc[3] * iw;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Vektor-Visualisierung
// ---------------------------------------------------------------------------

void VisualizeFlow(const Flow& flow, int flowScale, int outW, int outH, RGBAImage& dst)
{
    dst.resize(outW, outH);
    if (flow.empty()) {
        for (size_t i = 3; i < dst.d.size(); i += 4) dst.d[i] = 1.f;
        return;
    }
    const float scale = static_cast<float>(std::max(1, flowScale));
    const float inv = 1.f / scale;
    const float maxLen = 30.f; // Pixel, bei denen volle Helligkeit erreicht wird

    ParallelFor(0, outH, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < outW; ++x) {
                float fx = (x + 0.5f) * inv - 0.5f, fy = (y + 0.5f) * inv - 0.5f;
                float u = Sample(flow.u, fx, fy) * scale;
                float v = Sample(flow.v, fx, fy) * scale;
                float mag = std::min(1.f, std::sqrt(u * u + v * v) / maxLen);
                float hue = (std::atan2(v, u) + 3.14159265f) / (2.f * 3.14159265f) * 6.f;
                int   hi = static_cast<int>(hue) % 6;
                float f = hue - std::floor(hue);
                float q = 1 - f, t = f;
                float r, g, b;
                switch (hi) {
                    case 0: r = 1; g = t; b = 0; break;
                    case 1: r = q; g = 1; b = 0; break;
                    case 2: r = 0; g = 1; b = t; break;
                    case 3: r = 0; g = q; b = 1; break;
                    case 4: r = t; g = 0; b = 1; break;
                    default: r = 1; g = 0; b = q; break;
                }
                float* o = dst.px(x, y);
                o[0] = r * mag; o[1] = g * mag; o[2] = b * mag; o[3] = 1.f;
            }
    });
}


// ---------------------------------------------------------------------------
// Startfeld fuer die groebste Stufe
// ---------------------------------------------------------------------------

void CoarseInit(const FlowParams& p, int I0w, int w, int h, Flow& out)
{
    out.u.resize(w, h);
    out.v.resize(w, h);
    const float ls = static_cast<float>(w) / std::max(1, I0w);
    if (p.initFlow && !p.initFlow->empty()) {
        const Flow& f = *p.initFlow;
        const float sx = static_cast<float>(f.w()) / w, sy = static_cast<float>(f.h()) / h;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float fx = (x + 0.5f) * sx - 0.5f, fy = (y + 0.5f) * sy - 0.5f;
                out.u.row(y)[x] = Sample(f.u, fx, fy) * ls;
                out.v.row(y)[x] = Sample(f.v, fx, fy) * ls;
            }
    } else {
        std::fill(out.u.d.begin(), out.u.d.end(), p.initU * ls);
        std::fill(out.v.d.begin(), out.v.d.end(), p.initV * ls);
    }
}

// ---------------------------------------------------------------------------
// Kamerabewegung direkt aus den Bildern (parametrisches Lucas-Kanade)
// ---------------------------------------------------------------------------
// Normierte Koordinaten: xn = (x - cx) / s0 mit s0 = max(w,h)/2. Dadurch ist H
// unabhaengig von der Pyramidenstufe.

static inline bool MapH(const double H[9], double xn, double yn, double& un, double& vn, double& z)
{
    z = H[6] * xn + H[7] * yn + 1.0;
    if (z < 0.05) return false; // jenseits des Horizonts der Ebene
    un = (H[0] * xn + H[1] * yn + H[2]) / z;
    vn = (H[3] * xn + H[4] * yn + H[5]) / z;
    return true;
}

// Physikalisch moegliche Kamerabewegung zwischen zwei Frames? Verwirft z. B.
// Loesungen, die das Bild auf einen Punkt schrumpfen oder extrem verzerren.
static bool PlausibleH(const double h[8])
{
    for (int i = 0; i < 8; ++i) if (!std::isfinite(h[i])) return false;
    const double det = h[0] * h[4] - h[1] * h[3];
    if (det < 0.2 || det > 5.0) return false;              // Flaechenaenderung max. Faktor 5
    if (std::fabs(h[1]) > 1.0 || std::fabs(h[3]) > 1.0) return false;
    if (std::fabs(h[6]) > 1.5 || std::fabs(h[7]) > 1.5) return false;
    if (std::fabs(h[2]) > 2.0 || std::fabs(h[5]) > 2.0) return false; // Verschiebung max. 1 Bildbreite
    return true;
}

// Optimiert h (8 Parameter) auf einer Pyramidenstufe. Rueckgabe: mittlerer
// Restfehler inkl. Strafterm fuer kleine Ueberdeckung (kleiner = besser).
static double OptimizeLevel(const Image& A, const Image& B, const Image& Bx, const Image& By,
                            double h[8], int iters)
{
    const int w = A.w, hh = A.h;
    const double s0 = 0.5 * std::max(w, hh), cx = 0.5 * (w - 1), cy = 0.5 * (hh - 1);
    const int step = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(w) * hh / 40000.0)));
    int total = 0;
    for (int y = 1; y < hh - 1; y += step) for (int x = 1; x < w - 1; x += step) ++total;
    std::vector<float> res;
    double cost = 1e9;
    for (int it = 0; it <= iters; ++it) {
        const double Hc[9] = { h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0 };
        res.clear();
        for (int y = 1; y < hh - 1; y += step)
            for (int x = 1; x < w - 1; x += step) {
                double xn = (x - cx) / s0, yn = (y - cy) / s0, un, vn, z;
                if (!MapH(Hc, xn, yn, un, vn, z)) continue;
                double px = un * s0 + cx, py = vn * s0 + cy;
                if (px < 1 || py < 1 || px > w - 2 || py > hh - 2) continue;
                res.push_back(std::fabs(Sample(B, (float)px, (float)py) - A.row(y)[x]));
            }
        const double coverage = total ? static_cast<double>(res.size()) / total : 0.0;
        if (res.size() < 30) return 1e9;
        double mean = 0; for (float r : res) mean += r; mean /= res.size();
        cost = mean + 0.1 * std::max(0.0, 0.6 - coverage);
        if (it == iters) break;
        std::nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
        const double k = std::max(0.01, 1.4826 * res[res.size() / 2] * 1.5);

        std::vector<double> AtA(64, 0.0), Atb(8, 0.0);
        for (int y = 1; y < hh - 1; y += step)
            for (int x = 1; x < w - 1; x += step) {
                double xn = (x - cx) / s0, yn = (y - cy) / s0, un, vn, z;
                if (!MapH(Hc, xn, yn, un, vn, z)) continue;
                double px = un * s0 + cx, py = vn * s0 + cy;
                if (px < 1 || py < 1 || px > w - 2 || py > hh - 2) continue;
                double r = Sample(B, (float)px, (float)py) - A.row(y)[x];
                double gx = Sample(Bx, (float)px, (float)py) * s0, gy = Sample(By, (float)px, (float)py) * s0;
                double iz = 1.0 / z, gw = gx * un + gy * vn;
                double J[8] = { gx * xn * iz, gx * yn * iz, gx * iz, gy * xn * iz, gy * yn * iz, gy * iz,
                                -gw * xn * iz, -gw * yn * iz };
                double ar = std::fabs(r), wgt = ar <= k ? 1.0 : k / ar;
                for (int a2 = 0; a2 < 8; ++a2) {
                    Atb[a2] -= wgt * J[a2] * r;
                    for (int b2 = a2; b2 < 8; ++b2) AtA[a2 * 8 + b2] += wgt * J[a2] * J[b2];
                }
            }
        for (int a2 = 0; a2 < 8; ++a2) for (int b2 = 0; b2 < a2; ++b2) AtA[a2 * 8 + b2] = AtA[b2 * 8 + a2];
        for (int a2 = 0; a2 < 8; ++a2) AtA[a2 * 8 + a2] = AtA[a2 * 8 + a2] * 1.001 + 1e-9;
        if (!SolveLinear(AtA, Atb, 8)) return 1e9;
        double maxD = 0;
        for (int a2 = 0; a2 < 8; ++a2) {
            double d = std::max(-0.2, std::min(0.2, Atb[a2]));
            h[a2] += d; maxD = std::max(maxD, std::fabs(d));
        }
        if (maxD < 1e-6) { iters = it + 1; }
        if (!PlausibleH(h)) return 1e9;
    }
    for (int i = 0; i < 8; ++i) if (!std::isfinite(h[i])) return 1e9;
    if (!PlausibleH(h)) return 1e9;
    return cost;
}

double HomographyCost(const Image& I0, const Image& I1, const double H[9])
{
    if (I0.empty() || I0.w != I1.w || I0.h != I1.h) return 1e9;
    // auf ~480 px verkleinert bewerten (wie die feinste Stufe von AlignHomography)
    Image A, B, sa, sb;
    BoxFilter(I0, 1, A); BoxFilter(I1, 1, B);
    while (std::max(A.w, A.h) > 480) { Downscale(A, 2, sa); Downscale(B, 2, sb); BoxFilter(sa, 1, A); BoxFilter(sb, 1, B); }
    Image Bx, By; Bx.resize(B.w, B.h); By.resize(B.w, B.h);
    double h[8]; for (int k = 0; k < 8; ++k) h[k] = H[k] / H[8];
    return OptimizeLevel(A, B, Bx, By, h, 0);
}

bool InvertHomography(const double H[9], double Hi[9])
{
    const double a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7], i = H[8];
    double A[9] = { e * i - f * h, c * h - b * i, b * f - c * e,
                    f * g - d * i, a * i - c * g, c * d - a * f,
                    d * h - e * g, b * g - a * h, a * e - b * d };
    const double det = a * A[0] + b * A[3] + c * A[6];
    if (std::fabs(det) < 1e-12 || std::fabs(A[8]) < 1e-12) return false;
    for (int k = 0; k < 9; ++k) Hi[k] = A[k] / A[8];
    return true;
}

bool AlignHomography(const Image& I0, const Image& I1, float initDx, float initDy,
                     double H[9], double* residual, const double* extraInit)
{
    if (I0.empty() || I0.w != I1.w || I0.h != I1.h) return false;
    // Pyramide bis ~48 px
    std::vector<Image> P0(1), P1(1);
    BoxFilter(I0, 1, P0[0]); BoxFilter(I1, 1, P1[0]);
    while (std::max(P0.back().w, P0.back().h) > 64 && std::min(P0.back().w, P0.back().h) / 2 >= 24) {
        Image a, b, sa, sb;
        Downscale(P0.back(), 2, a); Downscale(P1.back(), 2, b);
        BoxFilter(a, 1, sa); BoxFilter(b, 1, sb);
        P0.push_back(std::move(sa)); P1.push_back(std::move(sb));
    }
    int finest = 0;
    while (finest + 1 < static_cast<int>(P0.size()) && std::max(P0[finest].w, P0[finest].h) > 480) ++finest;
    const int coarsest = static_cast<int>(P0.size()) - 1;

    auto grads = [](const Image& B, Image& Bx, Image& By) {
        const int w = B.w, hh = B.h;
        Bx.resize(w, hh); By.resize(w, hh);
        for (int y = 0; y < hh; ++y)
            for (int x = 0; x < w; ++x) {
                Bx.row(y)[x] = 0.5f * (B.row(y)[std::min(x + 1, w - 1)] - B.row(y)[std::max(x - 1, 0)]);
                By.row(y)[x] = 0.5f * (B.row(std::min(y + 1, hh - 1))[x] - B.row(std::max(y - 1, 0))[x]);
            }
    };

    // Startlagen: Verschiebung aus der Vorsuche, kombiniert mit Vorwaerts-/
    // Rueckwaertsflug (Zoom) um verschiedene Fluchtpunkte. Grosse Bewegungen
    // (Speed Ramps) liegen sonst ausserhalb des Einzugsbereichs.
    const double s00 = 0.5 * std::max(I0.w, I0.h);
    const double tx = initDx / s00, ty = initDy / s00;
    struct Hyp { double h[8]; double cost; };
    std::vector<Hyp> hyps;
    const double scales[] = { 1.0, 1.08, 1.18, 1.32, 1.5, 1.75, 0.93, 0.85, 0.76, 0.66 };
    const double centersY[] = { 0.0, -0.5, -0.9, 0.5 };
    for (double sc : scales)
        for (double cyn : centersY) {
            if (sc == 1.0 && cyn != 0.0) continue;
            // Skalierung um (0, cyn): x' = sc*x, y' = cyn + sc*(y - cyn)
            Hyp hp = { { sc, 0, tx, 0, sc, ty + cyn * (1 - sc), 0, 0 }, 0 };
            hyps.push_back(hp);
        }

    {
        Image Bx, By; grads(P1[coarsest], Bx, By);
        for (Hyp& hp : hyps) hp.cost = OptimizeLevel(P0[coarsest], P1[coarsest], Bx, By, hp.h, 30);
    }
    std::sort(hyps.begin(), hyps.end(), [](const Hyp& a, const Hyp& b) { return a.cost < b.cost; });
    // die besten drei auf der naechsten Stufe gegeneinander pruefen
    if (hyps.size() > 3) hyps.resize(3);
    int level = coarsest;
    if (coarsest - 1 >= finest) {
        level = coarsest - 1;
        Image Bx, By; grads(P1[level], Bx, By);
        for (Hyp& hp : hyps) hp.cost = OptimizeLevel(P0[level], P1[level], Bx, By, hp.h, 20);
        std::sort(hyps.begin(), hyps.end(), [](const Hyp& a, const Hyp& b) { return a.cost < b.cost; });
    }
    double h[8]; std::copy(hyps[0].h, hyps[0].h + 8, h);
    double lastCost = hyps[0].cost;
    for (int L = level - 1; L >= finest; --L) {
        Image Bx, By; grads(P1[L], Bx, By);
        lastCost = OptimizeLevel(P0[L], P1[L], Bx, By, h, 20);
    }
    // Zusaetzliche Startlage (z. B. Umkehrung der Gegenrichtung): nur auf den
    // feinen Stufen verfeinern - grobe Stufen koennten sie verwerfen.
    if (extraInit && std::fabs(extraInit[8]) > 1e-12) {
        double he[8]; for (int k = 0; k < 8; ++k) he[k] = extraInit[k] / extraInit[8];
        double ce = 1e9;
        for (int L = std::min(coarsest, finest + 1); L >= finest; --L) {
            Image Bx, By; grads(P1[L], Bx, By);
            ce = OptimizeLevel(P0[L], P1[L], Bx, By, he, 20);
        }
        if (ce < lastCost) { std::copy(he, he + 8, h); lastCost = ce; }
    }
    if (lastCost >= 1e8) return false;
    for (int i = 0; i < 8; ++i) H[i] = h[i];
    H[8] = 1.0;
    if (residual) *residual = lastCost;
    return true;
}

void HomographyField(const double H[9], int w, int h, Flow& out)
{
    out.u.resize(w, h);
    out.v.resize(w, h);
    const double s0 = 0.5 * std::max(w, h), cx = 0.5 * (w - 1), cy = 0.5 * (h - 1);
    const float lim = static_cast<float>(std::max(w, h));
    ParallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                double xn = (x - cx) / s0, yn = (y - cy) / s0, un, vn, z;
                float u = 0, v = 0;
                if (MapH(H, xn, yn, un, vn, z)) {
                    u = static_cast<float>(un * s0 + cx - x);
                    v = static_cast<float>(vn * s0 + cy - y);
                    if (std::fabs(u) > lim || std::fabs(v) > lim) u = v = 0; // unbrauchbar
                }
                out.u.row(y)[x] = u; out.v.row(y)[x] = v;
            }
    });
}

// ---------------------------------------------------------------------------
// Auswahl je Bildstelle: Modell / feines Tracking / keine Bewegung
// ---------------------------------------------------------------------------

void SelectCandidates(const Image& I0, const Image* Inext, const Image* Iprev,
                      Flow* fwd, Flow* bwd, const Flow* mf, const Flow* mb, SelectStats* stats)
{
    const int w = I0.w, h = I0.h;
    if (fwd && fwd->empty()) fwd = nullptr;
    if (bwd && bwd->empty()) bwd = nullptr;
    if (!fwd) Inext = nullptr;
    if (!bwd) Iprev = nullptr;
    if (!Inext && !Iprev) return;
    if (mf && mf->empty()) mf = nullptr;
    if (mb && mb->empty()) mb = nullptr;

    // Fotometrischer Fehler eines Kandidaten (beide Richtungen zusammen)
    auto cost = [&](const Flow* cf, const Flow* cb, Image& out) {
        Image raw; raw.resize(w, h);
        ParallelFor(0, h, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < w; ++x) {
                    // Mittel ueber die Richtungen, deren Ziel im Bild liegt. Zeigt die
                    // Bewegung aus dem Bild (Inhalt verlaesst das Bild, z. B. unten beim
                    // Vorwaertsflug), zaehlt nur die andere Richtung - kein Strafwert.
                    float e = 0; int n = 0;
                    const float c0 = I0.row(y)[x];
                    if (Inext) {
                        float u = cf ? cf->u.row(y)[x] : 0.f, v = cf ? cf->v.row(y)[x] : 0.f;
                        float tx = x + u, ty = y + v;
                        if (tx >= 0 && ty >= 0 && tx <= w - 1 && ty <= h - 1) { e += std::fabs(Sample(*Inext, tx, ty) - c0); ++n; }
                    }
                    if (Iprev) {
                        float u = cb ? cb->u.row(y)[x] : 0.f, v = cb ? cb->v.row(y)[x] : 0.f;
                        float tx = x + u, ty = y + v;
                        if (tx >= 0 && ty >= 0 && tx <= w - 1 && ty <= h - 1) { e += std::fabs(Sample(*Iprev, tx, ty) - c0); ++n; }
                    }
                    raw.row(y)[x] = n ? e / n : 0.3f; // fuehrt in beide Richtungen aus dem Bild
                }
        });
        SoftFilter(raw, 6, out);
    };

    Image cLocal, cZero, cModel;
    cost(fwd, bwd, cLocal);
    cost(nullptr, nullptr, cZero);
    const bool haveModel = (Inext ? mf != nullptr : true) && (Iprev ? mb != nullptr : true);
    if (haveModel) cost(mf, mb, cModel);

    // Modell leicht bevorzugen (glatt, flackerfrei); "keine Bewegung" nur, wenn
    // es deutlich besser passt (z. B. Himmel, unendlich weit entfernt).
    long nM = 0, nL = 0, nZ = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float l = cLocal.row(y)[x], z = cZero.row(y)[x], m = haveModel ? cModel.row(y)[x] * 0.55f : 1e9f;
            int pick = 1; float best = l;
            if (m <= best) { best = m; pick = 0; }
            if (z < best * 0.5f) { best = z; pick = 2; }
            if (pick == 0) {
                ++nM;
                if (fwd && mf) { fwd->u.row(y)[x] = mf->u.row(y)[x]; fwd->v.row(y)[x] = mf->v.row(y)[x]; }
                if (bwd && mb) { bwd->u.row(y)[x] = mb->u.row(y)[x]; bwd->v.row(y)[x] = mb->v.row(y)[x]; }
            } else if (pick == 2) {
                ++nZ;
                if (fwd) { fwd->u.row(y)[x] = 0; fwd->v.row(y)[x] = 0; }
                if (bwd) { bwd->u.row(y)[x] = 0; bwd->v.row(y)[x] = 0; }
            } else ++nL;
        }
    // Uebergaenge zwischen den Kandidaten leicht glaetten
    for (Flow* f : { fwd, bwd }) {
        if (!f) continue;
        Image a, b; SoftFilter(f->u, 2, a); SoftFilter(f->v, 2, b);
        f->u = std::move(a); f->v = std::move(b);
    }
    if (stats) {
        const float n = static_cast<float>(std::max<long>(1, static_cast<long>(w) * h));
        stats->model = nM / n; stats->local = nL / n; stats->zero = nZ / n;
    }
}


void CameraFlows(const Image& I0, const Image* In, const Image* Ip, const FlowParams& p,
                 const FlowFn& flowFn, Flow* fwd, Flow* bwd, CameraInfo* info)
{
    CameraInfo ci;
    Flow mf, mb;
    double Hf[9], Hb[9];
    float fdx = 0, fdy = 0, bdx = 0, bdy = 0;
    if (In && fwd) { EstimateGlobalShift(I0, *In, fdx, fdy); ci.modelF = AlignHomography(I0, *In, fdx, fdy, Hf, &ci.resF); ci.whyF = ci.modelF ? "OK" : "SCHAETZUNG GESCHEITERT"; }
    if (Ip && bwd) { EstimateGlobalShift(I0, *Ip, bdx, bdy); ci.modelB = AlignHomography(I0, *Ip, bdx, bdy, Hb, &ci.resB); ci.whyB = ci.modelB ? "OK" : "SCHAETZUNG GESCHEITERT"; }
    ci.shiftFx = fdx; ci.shiftFy = fdy; ci.shiftBx = bdx; ci.shiftBy = bdy;
    const double Hid[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const double zeroF = (In && fwd) ? HomographyCost(I0, *In, Hid) : 0;
    const double zeroB = (Ip && bwd) ? HomographyCost(I0, *Ip, Hid) : 0;
    // Gegenrichtung als zusaetzliche Startlage: rueckwaerts ~ Umkehrung von vorwaerts.
    // Hilft besonders bei grossen Bewegungen, wo eine Richtung viel Inhalt verliert.
    if (ci.modelF && Ip && bwd) {
        double inv[9], H2[9], r2;
        if (InvertHomography(Hf, inv) && AlignHomography(I0, *Ip, bdx, bdy, H2, &r2, inv) && (!ci.modelB || r2 < ci.resB)) {
            std::copy(H2, H2 + 9, Hb); ci.resB = r2; ci.modelB = true;
        }
    }
    if (ci.modelB && In && fwd) {
        double inv[9], H2[9], r2;
        if (InvertHomography(Hb, inv) && AlignHomography(I0, *In, fdx, fdy, H2, &r2, inv) && (!ci.modelF || r2 < ci.resF)) {
            std::copy(H2, H2 + 9, Hf); ci.resF = r2; ci.modelF = true;
        }
    }
    // Modell nur verwenden, wenn es besser zu den Bildern passt als "keine Bewegung"
    ci.zeroF = zeroF; ci.zeroB = zeroB;
    if (ci.modelF) { ci.costF = HomographyCost(I0, *In, Hf); ci.whyF = "OK"; std::copy(Hf, Hf + 9, ci.Hf);
                     if (ci.costF >= zeroF) { ci.modelF = false; ci.whyF = "SCHLECHTER ALS RUHE"; } }
    if (ci.modelB) { ci.costB = HomographyCost(I0, *Ip, Hb); ci.whyB = "OK"; std::copy(Hb, Hb + 9, ci.Hb);
                     if (ci.costB >= zeroB) { ci.modelB = false; ci.whyB = "SCHLECHTER ALS RUHE"; } }
    auto track = [&](const Image* In1, Flow* out, Flow& model, bool ok, const double* Hm, float dx, float dy) {
        if (!In1 || !out) return;
        FlowParams q = p;
        if (ok) { HomographyField(Hm, I0.w, I0.h, model); q.initFlow = &model; }
        else    { q.initU = dx; q.initV = dy; }
        flowFn(I0, *In1, q, *out);
    };
    track(In, fwd, mf, ci.modelF, Hf, fdx, fdy);
    track(Ip, bwd, mb, ci.modelB, Hb, bdx, bdy);
    RefineFlows(I0, In ? fwd : nullptr, Ip ? bwd : nullptr);
    SelectCandidates(I0, In, Ip, In ? fwd : nullptr, Ip ? bwd : nullptr,
                     ci.modelF ? &mf : nullptr, ci.modelB ? &mb : nullptr, &ci.sel);
    if (info) *info = ci;
}

} // namespace fmb
