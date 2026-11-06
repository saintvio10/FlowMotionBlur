// test_metal.mm - vergleicht die Metal-Kerne (Premiere-GPU-Weg) mit der CPU.
// Ohne Metal-Geraet (z. B. virtueller Mac) wird der Test uebersprungen.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "FlowCore.h"
#include "MetalBackend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace fmb;

static uint16_t ToHalf(float f)
{
    __fp16 h = (__fp16)f; uint16_t r; std::memcpy(&r, &h, 2); return r;
}
static float FromHalf(uint16_t v)
{
    __fp16 h; std::memcpy(&h, &v, 2); return (float)h;
}

// Testbild (nicht vormultipliziert, Alpha teils < 1)
static void MakeImage(int w, int h, std::vector<float>& rgba)
{
    rgba.resize((size_t)w * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float* p = &rgba[((size_t)y * w + x) * 4];
            p[0] = 0.5f + 0.5f * std::sin(x * 0.21f + y * 0.05f);
            p[1] = 0.5f + 0.5f * std::cos(y * 0.17f - x * 0.03f);
            p[2] = ((x / 7 + y / 5) & 1) ? 0.9f : 0.1f;
            p[3] = (x < w / 5) ? 0.25f + 0.5f * (float)y / h : 1.f;
        }
}

static id<MTLBuffer> Upload(id<MTLDevice> dev, const std::vector<float>& rgba, int w, int h, int rowBytes, bool half)
{
    std::vector<uint8_t> bytes((size_t)rowBytes * h, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* s = &rgba[((size_t)y * w + x) * 4];
            const float bgra[4] = { s[2], s[1], s[0], s[3] };
            if (half) { uint16_t* d = (uint16_t*)(bytes.data() + (size_t)y * rowBytes) + x * 4; for (int k = 0; k < 4; ++k) d[k] = ToHalf(bgra[k]); }
            else      { float* d = (float*)(bytes.data() + (size_t)y * rowBytes) + x * 4; for (int k = 0; k < 4; ++k) d[k] = bgra[k]; }
        }
    return [dev newBufferWithBytes:bytes.data() length:bytes.size() options:MTLResourceStorageModeShared];
}

static void Download(id<MTLBuffer> b, int w, int h, int rowBytes, bool half, std::vector<float>& rgba)
{
    rgba.resize((size_t)w * h * 4);
    const uint8_t* base = (const uint8_t*)b.contents;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float bgra[4];
            if (half) { const uint16_t* s = (const uint16_t*)(base + (size_t)y * rowBytes) + x * 4; for (int k = 0; k < 4; ++k) bgra[k] = FromHalf(s[k]); }
            else      { const float* s = (const float*)(base + (size_t)y * rowBytes) + x * 4; for (int k = 0; k < 4; ++k) bgra[k] = s[k]; }
            float* d = &rgba[((size_t)y * w + x) * 4];
            d[0] = bgra[2]; d[1] = bgra[1]; d[2] = bgra[0]; d[3] = bgra[3];
        }
}

// Bild so runden, wie es im Puffer steht (fuer den Half-Vergleich)
static std::vector<float> Quantize(const std::vector<float>& v, bool half)
{
    if (!half) return v;
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = FromHalf(ToHalf(v[i]));
    return o;
}

static RGBAImage Premultiplied(const std::vector<float>& rgba, int w, int h)
{
    RGBAImage im; im.resize(w, h);
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        const float a = rgba[i * 4 + 3];
        im.d[i * 4 + 0] = rgba[i * 4 + 0] * a; im.d[i * 4 + 1] = rgba[i * 4 + 1] * a;
        im.d[i * 4 + 2] = rgba[i * 4 + 2] * a; im.d[i * 4 + 3] = a;
    }
    return im;
}

int main()
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("METAL: kein Geraet - Test uebersprungen\n"); return 0; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        std::printf("METAL: %s\n", [[dev name] UTF8String]);

        const int W = 157, H = 93, scale = 2;
        std::vector<float> img;
        MakeImage(W, H, img);

        // Vektorfelder in halber Aufloesung (Drehung + Verschiebung, mit Bewegung > Mindestbewegung)
        const int fw = (W + scale - 1) / scale, fh = (H + scale - 1) / scale;
        Flow fwd, bwd;
        fwd.u.resize(fw, fh); fwd.v.resize(fw, fh); bwd.u.resize(fw, fh); bwd.v.resize(fw, fh);
        for (int y = 0; y < fh; ++y)
            for (int x = 0; x < fw; ++x) {
                const float cx = x - fw * 0.5f, cy = y - fh * 0.5f;
                fwd.u.row(y)[x] = 3.0f + 0.08f * cy;   fwd.v.row(y)[x] = -1.0f - 0.08f * cx;
                bwd.u.row(y)[x] = -2.6f - 0.07f * cy;  bwd.v.row(y)[x] = 1.2f + 0.07f * cx;
            }

        int fails = 0;
        for (int half = 0; half <= 1; ++half) {
            const int bpp = half ? 8 : 16;
            const int rowBytes = W * bpp + 64; // mit Zeilen-Padding wie bei Premiere
            const std::vector<float> ref = Quantize(img, half);
            const RGBAImage pm = Premultiplied(ref, W, H);
            const double tol = half ? 4e-3 : 2e-5;

            // --- Luminanz ---
            for (int factor : { 1, 2, 4 }) {
                for (float aw : { 0.f, 0.5f }) {
                    id<MTLBuffer> b = Upload(dev, img, W, H, rowBytes, half);
                    metal::Frame f; f.buffer = (__bridge void*)b; f.w = W; f.h = H; f.rowBytes = rowBytes; f.isHalf = half;
                    Image lg; double am = 0;
                    if (!metal::Luma((__bridge void*)dev, (__bridge void*)q, f, factor, aw, lg, am)) {
                        std::printf("FEHLER Luma: %s\n", metal::LastError().c_str()); return 1;
                    }
                    Image lc; LumaFromRGBA(pm, factor, lc, aw);
                    double md = 0;
                    if (lg.w != lc.w || lg.h != lc.h) md = 1;
                    else for (size_t i = 0; i < lc.d.size(); ++i) md = std::max(md, (double)std::fabs(lg.d[i] - lc.d[i]));
                    double ac = 0; for (size_t i = 0; i < (size_t)W * H; ++i) ac += pm.d[i * 4 + 3]; ac /= (double)W * H;
                    const bool ok = md < tol && std::fabs(am - ac) < tol;
                    std::printf("LUMA %s faktor %d alpha %.1f: max. Abweichung %.2e, Alpha %.4f/%.4f %s\n",
                                half ? "16F" : "32F", factor, aw, md, am, ac, ok ? "OK" : "FEHLER");
                    if (!ok) ++fails;
                }
            }

            // --- Unschaerfe ---
            struct Case { const char* name; bool f, b; float shutter, minMotion; int maxS; };
            const Case cases[] = {
                { "vor+rueck", true, true, 0.5f, 0.f, 32 },
                { "nur vor", true, false, 1.0f, 0.f, 32 },
                { "nur rueck", false, true, 0.7f, 0.f, 16 },
                { "mindestbewegung", true, true, 2.0f, 5.f, 64 },
                { "stark", true, true, 4.0f, 0.f, 9 },
            };
            for (const Case& c : cases) {
                BlurParams bp; bp.shutter = c.shutter; bp.minMotion = c.minMotion; bp.maxSamples = c.maxS; bp.mode = kBlurNatural;
                RGBAImage out; out.resize(W, H);
                ApplyMotionBlur(pm, c.f ? &fwd : nullptr, c.b ? &bwd : nullptr, scale, bp, out);
                id<MTLBuffer> b = Upload(dev, img, W, H, rowBytes, half);
                metal::Frame f; f.buffer = (__bridge void*)b; f.w = W; f.h = H; f.rowBytes = rowBytes; f.isHalf = half;
                if (!metal::BlurNaturalInPlace((__bridge void*)dev, (__bridge void*)q, f, c.f ? &fwd : nullptr,
                                               c.b ? &bwd : nullptr, scale, bp)) {
                    std::printf("FEHLER Blur: %s\n", metal::LastError().c_str()); return 1;
                }
                std::vector<float> got; Download(b, W, H, rowBytes, half, got);
                // CPU-Ergebnis zurueck auf nicht vormultipliziert (wie der Software-Weg schreibt)
                double md = 0, mean = 0; size_t n = 0;
                for (size_t i = 0; i < (size_t)W * H; ++i) {
                    const float a = out.d[i * 4 + 3];
                    for (int k = 0; k < 4; ++k) {
                        float r = out.d[i * 4 + k];
                        if (k < 3 && a > 1e-6f) r /= a;
                        const double d = std::fabs(r - got[i * 4 + k]);
                        md = std::max(md, d); mean += d; ++n;
                    }
                }
                mean /= n;
                // Half: Rundung des Ergebnisses + Rechnung in float auf der GPU
                const double tolB = half ? 6e-3 : 5e-4;
                const bool ok = md < tolB;
                std::printf("BLUR %s %-16s max. Abweichung %.2e (Mittel %.2e) %s\n",
                            half ? "16F" : "32F", c.name, md, mean, ok ? "OK" : "FEHLER");
                if (!ok) ++fails;
            }

            // --- Ohne Bewegung: Bild bleibt unveraendert ---
            {
                Flow z; z.u.resize(fw, fh); z.v.resize(fw, fh);
                BlurParams bp;
                id<MTLBuffer> b = Upload(dev, img, W, H, rowBytes, half);
                std::vector<uint8_t> before((const uint8_t*)b.contents, (const uint8_t*)b.contents + b.length);
                metal::Frame f; f.buffer = (__bridge void*)b; f.w = W; f.h = H; f.rowBytes = rowBytes; f.isHalf = half;
                const bool ok = metal::BlurNaturalInPlace((__bridge void*)dev, (__bridge void*)q, f, &z, &z, scale, bp) &&
                                std::memcmp(before.data(), b.contents, before.size()) == 0;
                std::printf("BLUR %s ohne Bewegung unveraendert: %s\n", half ? "16F" : "32F", ok ? "OK" : "FEHLER");
                if (!ok) ++fails;
            }
        }
        std::printf(fails ? "METAL-TEST: %d FEHLER\n" : "METAL-TEST: ALLES OK\n", fails);
        return fails ? 1 : 0;
    }
}
