// MetalBackend.mm - Implementierung der Metal-Rechenkerne (macOS).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalBackend.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <vector>

namespace fmb { namespace metal {

// ---------------------------------------------------------------------------
// Rechenkerne (Metal Shading Language), werden zur Laufzeit kompiliert.
// Gleiche Rechnung wie src/core/FlowCore.cpp (LumaFromRGBA, ApplyMotionBlur
// im Modus "Natuerlich"), damit CPU- und GPU-Weg dasselbe Ergebnis liefern.
// ---------------------------------------------------------------------------
static const char* kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct LumaP { int w, h, rowBytes, isHalf, factor, ow, oh; float alphaW; };
struct BlurP {
    int w, h, rowIn, rowOut, halfIn, halfOut;
    int fw, fh, mode, maxS, soft;
    float scale, shutter, maxRadius, minMotion, phase;
};

// Pixel lesen/schreiben (BGRA -> RGBA)
inline float4 ldpx(device const uchar* base, int rowBytes, int x, int y, int isHalf)
{
    if (isHalf) { device const half4* p = (device const half4*)(base + (long)y * rowBytes); half4 v = p[x]; return float4(v.z, v.y, v.x, v.w); }
    device const float4* p = (device const float4*)(base + (long)y * rowBytes); float4 v = p[x]; return float4(v.z, v.y, v.x, v.w);
}
inline void stpx(device uchar* base, int rowBytes, int x, int y, int isHalf, float4 c)
{
    float4 b = float4(c.z, c.y, c.x, c.w);
    if (isHalf) { device half4* p = (device half4*)(base + (long)y * rowBytes); p[x] = half4(b); }
    else        { device float4* p = (device float4*)(base + (long)y * rowBytes); p[x] = b; }
}
// Premiere liefert nicht vormultipliziertes Alpha; gerechnet wird (wie im
// CPU-Weg) vormultipliziert, damit transparente Kanten sauber bleiben.
inline float4 ldpm(device const uchar* base, int rowBytes, int x, int y, int isHalf)
{
    float4 c = ldpx(base, rowBytes, x, y, isHalf); return float4(c.xyz * c.w, c.w);
}
inline float4 smp4(device const uchar* base, int rowBytes, int w, int h, int isHalf, float x, float y)
{
    x = clamp(x, 0.0f, float(w - 1)); y = clamp(y, 0.0f, float(h - 1));
    int x0 = int(x), y0 = int(y);
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float fx = x - x0, fy = y - y0;
    return ldpm(base, rowBytes, x0, y0, isHalf) * ((1 - fx) * (1 - fy)) + ldpm(base, rowBytes, x1, y0, isHalf) * (fx * (1 - fy))
         + ldpm(base, rowBytes, x0, y1, isHalf) * ((1 - fx) * fy)       + ldpm(base, rowBytes, x1, y1, isHalf) * (fx * fy);
}
inline float smp(device const float* im, int w, int h, float x, float y)
{
    x = clamp(x, 0.0f, float(w - 1)); y = clamp(y, 0.0f, float(h - 1));
    int x0 = int(x), y0 = int(y);
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float fx = x - x0, fy = y - y0;
    float a = im[y0 * w + x0] + (im[y0 * w + x1] - im[y0 * w + x0]) * fx;
    float b = im[y1 * w + x0] + (im[y1 * w + x1] - im[y1 * w + x0]) * fx;
    return a + (b - a) * fy;
}

// Verkleinerte Luminanz (vormultipliziert wie im CPU-Weg) und Alpha
kernel void luma_down(device const uchar* src [[buffer(0)]], device float* outL [[buffer(1)]],
                      device float* outA [[buffer(2)]], constant LumaP& P [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= P.ow || (int)gid.y >= P.oh) return;
    int sy0 = gid.y * P.factor, sy1 = min(P.h, sy0 + P.factor);
    int sx0 = gid.x * P.factor, sx1 = min(P.w, sx0 + P.factor);
    float s = 0, sa = 0; int n = 0;
    for (int y = sy0; y < sy1; ++y)
        for (int x = sx0; x < sx1; ++x) {
            float4 c = ldpx(src, P.rowBytes, x, y, P.isHalf);
            float3 pm = c.xyz * c.w;
            float l = 0.299f * pm.x + 0.587f * pm.y + 0.114f * pm.z;
            s += l * (1.0f - P.alphaW) + c.w * P.alphaW; sa += c.w; ++n;
        }
    outL[gid.y * P.ow + gid.x] = s / n;
    outA[gid.y * P.ow + gid.x] = sa / n;
}

// Bewegungsunschaerfe "Natuerlich": jeder Pixel mittelt entlang seiner Bahn
// (wie MakeBlurPath/ApplyMotionBlur in FlowCore). mode: 0 = vor+rueck, 1 = vor, 2 = rueck
kernel void blur_natural(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                         device const float* fu [[buffer(2)]], device const float* fv [[buffer(3)]],
                         device const float* bu [[buffer(4)]], device const float* bv [[buffer(5)]],
                         constant BlurP& P [[buffer(6)]], uint2 gid [[thread_position_in_grid]])
{
    int x = gid.x, y = gid.y;
    if (x >= P.w || y >= P.h) return;
    float inv = 1.0f / P.scale;
    float fx = (x + 0.5f) * inv - 0.5f, fy = (y + 0.5f) * inv - 0.5f;
    float a = 0, b = 0, c = 0, d = 0;
    if (P.mode != 2) { a = smp(fu, P.fw, P.fh, fx, fy) * P.scale; b = smp(fv, P.fw, P.fh, fx, fy) * P.scale; }
    if (P.mode != 1) { c = smp(bu, P.fw, P.fh, fx, fy) * P.scale; d = smp(bv, P.fw, P.fh, fx, fy) * P.scale; }
    float vx, vy;
    if (P.mode == 0)      { vx = 0.5f * (a - c); vy = 0.5f * (b - d); }
    else if (P.mode == 1) { vx = a; vy = b; }
    else                  { vx = -c; vy = -d; }
    if (P.minMotion > 0.0f) {
        float l = sqrt(vx * vx + vy * vy);
        float k = l > P.minMotion ? (l - P.minMotion) / l : 0.0f;
        vx *= k; vy *= k;
    }
    float hs = 0.5f * P.shutter;
    float speed = sqrt(vx * vx + vy * vy);
    float l0 = speed * fabs(hs * (-1.0f + P.phase)), l1 = speed * fabs(hs * (1.0f + P.phase));
    float L = max(l0, l1);
    if (L > P.maxRadius) { float k = P.maxRadius / L; vx *= k; vy *= k; l0 *= k; l1 *= k; L = P.maxRadius; }
    if (L < 0.25f) { stpx(dst, P.rowOut, x, y, P.halfOut, ldpx(src, P.rowIn, x, y, P.halfIn)); return; }
    int S = min(P.maxS, int(ceil(l0 + l1)) + 1);
    if ((S & 1) == 0) ++S;
    float4 acc = float4(0.0f); float wsum = 0.0f;
    for (int i = 0; i < S; ++i) {
        float t = (S == 1) ? 0.0f : -1.0f + 2.0f * i / (S - 1);
        float w = P.soft ? clamp((1.0f - fabs(t)) / 0.35f, 0.0f, 1.0f) : 1.0f;
        if (w <= 0.0f) continue;
        float sp = hs * (t + P.phase);
        acc += w * smp4(src, P.rowIn, P.w, P.h, P.halfIn, x - vx * sp, y - vy * sp);
        wsum += w;
    }
    acc *= (1.0f / wsum);
    if (acc.w > 1e-6f) acc.xyz /= acc.w;
    stpx(dst, P.rowOut, x, y, P.halfOut, acc);
}

kernel void copy_px(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                    constant BlurP& P [[buffer(6)]], uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= P.w || (int)gid.y >= P.h) return;
    stpx(dst, P.rowOut, gid.x, gid.y, P.halfOut, ldpx(src, P.rowIn, gid.x, gid.y, P.halfIn));
}
)MSL";

struct LumaP { int w, h, rowBytes, isHalf, factor, ow, oh; float alphaW; };
struct BlurP {
    int w, h, rowIn, rowOut, halfIn, halfOut;
    int fw, fh, mode, maxS, soft;
    float scale, shutter, maxRadius, minMotion, phase;
};

struct Pipelines {
    id<MTLComputePipelineState> luma = nil, blur = nil, copy = nil;
};

static std::mutex gMutex;
static std::map<void*, Pipelines> gPipelines;
static std::string gLastError;

std::string LastError() { std::lock_guard<std::mutex> l(gMutex); return gLastError; }

static void SetError(const std::string& e) { gLastError = e; }

static bool GetPipelines(id<MTLDevice> dev, Pipelines& out)
{
    std::lock_guard<std::mutex> l(gMutex);
    auto it = gPipelines.find((__bridge void*)dev);
    if (it != gPipelines.end()) { out = it->second; return true; }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSource] options:nil error:&err];
    if (!lib) {
        SetError(std::string("Metal-Kompilierung: ") + (err ? [[err localizedDescription] UTF8String] : "?"));
        return false;
    }
    Pipelines p;
    auto make = [&](const char* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!f) return nil;
        NSError* e2 = nil;
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&e2];
        if (!ps) SetError(std::string("Metal-Pipeline ") + name + ": " + (e2 ? [[e2 localizedDescription] UTF8String] : "?"));
        return ps;
    };
    p.luma = make("luma_down"); p.blur = make("blur_natural"); p.copy = make("copy_px");
    if (!p.luma || !p.blur || !p.copy) return false;
    gPipelines[(__bridge void*)dev] = p;
    out = p;
    return true;
}

static void Dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, int w, int h)
{
    const NSUInteger tw = ps.threadExecutionWidth;
    const NSUInteger th = std::max<NSUInteger>(1, ps.maxTotalThreadsPerThreadgroup / tw);
    MTLSize tg = MTLSizeMake(tw, std::min<NSUInteger>(th, 16), 1);
    MTLSize groups = MTLSizeMake((w + tg.width - 1) / tg.width, (h + tg.height - 1) / tg.height, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
}

static bool Finish(id<MTLCommandBuffer> cb)
{
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        std::lock_guard<std::mutex> l(gMutex);
        SetError(std::string("Metal-Ausfuehrung: ") + (cb.error ? [[cb.error localizedDescription] UTF8String] : "Status " + std::to_string((int)cb.status)));
        return false;
    }
    return true;
}

bool Luma(void* device, void* queue, const Frame& in, int factor, float alphaWeight, Image& luma, double& alphaMean)
{
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLBuffer> src = (__bridge id<MTLBuffer>)in.buffer;
        if (!dev || !q || !src) return false;
        Pipelines P; if (!GetPipelines(dev, P)) return false;
        factor = std::max(1, factor);
        LumaP lp = { in.w, in.h, in.rowBytes, in.isHalf ? 1 : 0, factor, (in.w + factor - 1) / factor, (in.h + factor - 1) / factor, alphaWeight };
        const size_t n = (size_t)lp.ow * lp.oh;
        id<MTLBuffer> bl = [dev newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ba = [dev newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        if (!bl || !ba) return false;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:P.luma];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:bl offset:0 atIndex:1];
        [enc setBuffer:ba offset:0 atIndex:2];
        [enc setBytes:&lp length:sizeof(lp) atIndex:3];
        Dispatch(enc, P.luma, lp.ow, lp.oh);
        [enc endEncoding];
        if (!Finish(cb)) return false;
        luma.resize(lp.ow, lp.oh);
        std::copy((const float*)bl.contents, (const float*)bl.contents + n, luma.d.begin());
        const float* a = (const float*)ba.contents;
        double s = 0; for (size_t i = 0; i < n; ++i) s += a[i];
        alphaMean = n ? s / n : 0.0;
        return true;
    }
}

static BlurP MakeBlurP(const Frame& src, const Frame& dst)
{
    BlurP bp = {};
    bp.w = src.w; bp.h = src.h; bp.rowIn = src.rowBytes; bp.rowOut = dst.rowBytes;
    bp.halfIn = src.isHalf ? 1 : 0; bp.halfOut = dst.isHalf ? 1 : 0;
    return bp;
}

bool Copy(void* device, void* queue, const Frame& src, const Frame& dst)
{
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        if (!dev || !q || !src.buffer || !dst.buffer) return false;
        Pipelines P; if (!GetPipelines(dev, P)) return false;
        BlurP bp = MakeBlurP(src, dst);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:P.copy];
        [enc setBuffer:(__bridge id<MTLBuffer>)src.buffer offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst.buffer offset:0 atIndex:1];
        [enc setBytes:&bp length:sizeof(bp) atIndex:6];
        Dispatch(enc, P.copy, bp.w, bp.h);
        [enc endEncoding];
        return Finish(cb);
    }
}

bool BlurNatural(void* device, void* queue, const Frame& src, const Frame& dst,
                 const Flow* fwd, const Flow* bwd, int flowScale, const BlurParams& p)
{
    if (fwd && fwd->empty()) fwd = nullptr;
    if (bwd && bwd->empty()) bwd = nullptr;
    if ((!fwd && !bwd) || p.shutter <= 0.f) return Copy(device, queue, src, dst);
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        if (!dev || !q || !src.buffer || !dst.buffer) return false;
        Pipelines P; if (!GetPipelines(dev, P)) return false;
        const Flow& any = fwd ? *fwd : *bwd;
        BlurP bp = MakeBlurP(src, dst);
        bp.fw = any.w(); bp.fh = any.h();
        bp.mode = (fwd && bwd) ? 0 : (fwd ? 1 : 2);
        bp.maxS = std::max(1, p.maxSamples);
        bp.scale = (float)std::max(1, flowScale);
        bp.shutter = p.shutter;
        bp.maxRadius = kMaxBlurRadius;
        bp.minMotion = p.minMotion;
        bp.phase = p.phase;
        bp.soft = p.soft ? 1 : 0;
        auto upload = [&](const Image* im) -> id<MTLBuffer> {
            if (!im || im->d.empty()) return [dev newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
            return [dev newBufferWithBytes:im->d.data() length:im->d.size() * sizeof(float) options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> bfu = upload(fwd ? &fwd->u : nullptr), bfv = upload(fwd ? &fwd->v : nullptr);
        id<MTLBuffer> bbu = upload(bwd ? &bwd->u : nullptr), bbv = upload(bwd ? &bwd->v : nullptr);
        if (!bfu || !bfv || !bbu || !bbv) return false;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:P.blur];
        [enc setBuffer:(__bridge id<MTLBuffer>)src.buffer offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst.buffer offset:0 atIndex:1];
        [enc setBuffer:bfu offset:0 atIndex:2];
        [enc setBuffer:bfv offset:0 atIndex:3];
        [enc setBuffer:bbu offset:0 atIndex:4];
        [enc setBuffer:bbv offset:0 atIndex:5];
        [enc setBytes:&bp length:sizeof(bp) atIndex:6];
        Dispatch(enc, P.blur, bp.w, bp.h);
        [enc endEncoding];
        return Finish(cb);
    }
}

// Zwischenpuffer fuer die Unschaerfe direkt in Premieres Bild (wiederverwendet)
static std::mutex gScratchMutex;
static std::vector<std::pair<void*, id<MTLBuffer>>> gScratch; // (Geraet, Puffer)

static id<MTLBuffer> TakeScratch(id<MTLDevice> dev, NSUInteger len)
{
    {
        std::lock_guard<std::mutex> l(gScratchMutex);
        for (size_t i = 0; i < gScratch.size(); ++i)
            if (gScratch[i].first == (__bridge void*)dev && gScratch[i].second.length >= len) {
                id<MTLBuffer> b = gScratch[i].second; gScratch.erase(gScratch.begin() + i); return b;
            }
    }
    return [dev newBufferWithLength:len options:MTLResourceStorageModePrivate];
}
static void GiveScratch(id<MTLDevice> dev, id<MTLBuffer> b)
{
    if (!b) return;
    std::lock_guard<std::mutex> l(gScratchMutex);
    if (gScratch.size() >= 3) gScratch.erase(gScratch.begin());
    gScratch.emplace_back((__bridge void*)dev, b);
}

// Groesste Unschaerfe-Strecke (wie im CPU-Weg: unter 0.5 px bleibt das Bild unveraendert)
static float MaxBlurLength(const Flow* fwd, const Flow* bwd, int flowScale, const BlurParams& p)
{
    const Flow& any = fwd ? *fwd : *bwd;
    const float scale = (float)std::max(1, flowScale);
    float m = 0.f;
    for (size_t i = 0; i < any.u.d.size(); ++i) {
        const float fu = fwd ? fwd->u.d[i] * scale : 0.f, fv = fwd ? fwd->v.d[i] * scale : 0.f;
        const float bu = bwd ? bwd->u.d[i] * scale : 0.f, bv = bwd ? bwd->v.d[i] * scale : 0.f;
        m = std::max(m, MakeBlurPath(fwd != nullptr, bwd != nullptr, fu, fv, bu, bv, p, kMaxBlurRadius).extent);
    }
    return m;
}

bool BlurNaturalInPlace(void* device, void* queue, const Frame& frame,
                        const Flow* fwd, const Flow* bwd, int flowScale, const BlurParams& p)
{
    if (fwd && fwd->empty()) fwd = nullptr;
    if (bwd && bwd->empty()) bwd = nullptr;
    if ((!fwd && !bwd) || p.shutter <= 0.f) return true; // Bild bleibt, wie es ist
    if (fwd && bwd && (fwd->u.d.size() != bwd->u.d.size())) bwd = nullptr;
    if (MaxBlurLength(fwd, bwd, flowScale, p) < 0.5f) return true;
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)frame.buffer;
        if (!dev || !q || !buf) return false;
        Pipelines P; if (!GetPipelines(dev, P)) return false;
        const NSUInteger len = (NSUInteger)frame.rowBytes * (NSUInteger)frame.h;
        if (len == 0 || len > buf.length) { std::lock_guard<std::mutex> l(gMutex); SetError("Puffer zu klein"); return false; }
        id<MTLBuffer> scratch = TakeScratch(dev, len);
        if (!scratch) return false;
        const Flow& any = fwd ? *fwd : *bwd;
        BlurP bp = MakeBlurP(frame, frame);
        bp.fw = any.w(); bp.fh = any.h();
        bp.mode = (fwd && bwd) ? 0 : (fwd ? 1 : 2);
        bp.maxS = std::max(1, p.maxSamples);
        bp.scale = (float)std::max(1, flowScale);
        bp.shutter = p.shutter;
        bp.maxRadius = kMaxBlurRadius;
        bp.minMotion = p.minMotion;
        bp.phase = p.phase;
        bp.soft = p.soft ? 1 : 0;
        auto upload = [&](const Image* im) -> id<MTLBuffer> {
            if (!im || im->d.empty()) return [dev newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
            return [dev newBufferWithBytes:im->d.data() length:im->d.size() * sizeof(float) options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> bfu = upload(fwd ? &fwd->u : nullptr), bfv = upload(fwd ? &fwd->v : nullptr);
        id<MTLBuffer> bbu = upload(bwd ? &bwd->u : nullptr), bbv = upload(bwd ? &bwd->v : nullptr);
        if (!bfu || !bfv || !bbu || !bbv) { GiveScratch(dev, scratch); return false; }
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:buf sourceOffset:0 toBuffer:scratch destinationOffset:0 size:len];
        [blit endEncoding];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:P.blur];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:buf offset:0 atIndex:1];
        [enc setBuffer:bfu offset:0 atIndex:2];
        [enc setBuffer:bfv offset:0 atIndex:3];
        [enc setBuffer:bbu offset:0 atIndex:4];
        [enc setBuffer:bbv offset:0 atIndex:5];
        [enc setBytes:&bp length:sizeof(bp) atIndex:6];
        Dispatch(enc, P.blur, bp.w, bp.h);
        [enc endEncoding];
        const bool ok = Finish(cb);
        GiveScratch(dev, scratch);
        return ok;
    }
}

}} // namespace fmb::metal
