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
        } else if (guides) {
            // Track Points geben der groebsten Stufe einen Startwert,
            // damit auch grosse Bewegungen gefunden werden.
            ApplyGuides(init, *guides, p.guideRadius, static_cast<float>(w) / I0.w);
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

} // namespace fmb
