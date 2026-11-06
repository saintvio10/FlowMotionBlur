// Kernels.h - OpenCL-Kernel (OpenCL C 1.2), werden zur Laufzeit kompiliert.
//
// Jeder Kernel entspricht 1:1 einem Schritt in src/core/FlowCore.cpp.
// Aufgeteilt in mehrere String-Literale (MSVC erlaubt max. 16 KB pro Literal).

#pragma once

namespace fmb { namespace gpu {

static const char* const kKernelSourceParts[] = {
// ---------------------------------------------------------------- Hilfen
R"CLC(
inline float smp(__global const float* im, int w, int h, float x, float y)
{
    x = clamp(x, 0.0f, (float)(w - 1));
    y = clamp(y, 0.0f, (float)(h - 1));
    int x0 = (int)x, y0 = (int)y;
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float fx = x - x0, fy = y - y0;
    float a = im[y0 * w + x0] + (im[y0 * w + x1] - im[y0 * w + x0]) * fx;
    float b = im[y1 * w + x0] + (im[y1 * w + x1] - im[y1 * w + x0]) * fx;
    return a + (b - a) * fy;
}

inline float4 smp4(__global const float4* im, int w, int h, float x, float y)
{
    x = clamp(x, 0.0f, (float)(w - 1));
    y = clamp(y, 0.0f, (float)(h - 1));
    int x0 = (int)x, y0 = (int)y;
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float fx = x - x0, fy = y - y0;
    return im[y0 * w + x0] * ((1 - fx) * (1 - fy)) + im[y0 * w + x1] * (fx * (1 - fy))
         + im[y1 * w + x0] * ((1 - fx) * fy)       + im[y1 * w + x1] * (fx * fy);
}

// ---------------------------------------------------------------- Pyramide
__kernel void downscale(__global const float* src, int sw, int sh,
                        __global float* dst, int dw, int dh, int factor)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= dw || y >= dh) return;
    int sy0 = y * factor, sy1 = min(sh, sy0 + factor);
    int sx0 = x * factor, sx1 = min(sw, sx0 + factor);
    float s = 0; int n = 0;
    for (int sy = sy0; sy < sy1; ++sy)
        for (int sx = sx0; sx < sx1; ++sx) { s += src[sy * sw + sx]; ++n; }
    dst[y * dw + x] = s / n;
}

// ---------------------------------------------------------------- Gradienten
__kernel void gradients(__global const float* A, int w, int h,
                        __global float* Ix, __global float* Iy,
                        __global float* Ixx, __global float* Ixy, __global float* Iyy)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float gx = 0.5f * (A[y * w + min(x + 1, w - 1)] - A[y * w + max(x - 1, 0)]);
    float gy = 0.5f * (A[min(y + 1, h - 1) * w + x] - A[max(y - 1, 0) * w + x]);
    Ix[i] = gx; Iy[i] = gy;
    Ixx[i] = gx * gx; Ixy[i] = gx * gy; Iyy[i] = gy * gy;
}

// ---------------------------------------------------------------- Box-Filter
__kernel void box_h(__global const float* src, __global float* dst, int w, int h, int r)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int a = max(0, x - r), b = min(w - 1, x + r);
    float s = 0;
    for (int k = a; k <= b; ++k) s += src[y * w + k];
    dst[y * w + x] = s / (b - a + 1);
}

__kernel void box_v(__global const float* src, __global float* dst, int w, int h, int r)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int a = max(0, y - r), b = min(h - 1, y + r);
    float s = 0;
    for (int k = a; k <= b; ++k) s += src[k * w + x];
    dst[y * w + x] = s / (b - a + 1);
}
)CLC",
// ---------------------------------------------------------------- Lucas-Kanade
R"CLC(
__kernel void warp_products(__global const float* A, __global const float* B,
                            __global const float* u, __global const float* v,
                            __global const float* Ix, __global const float* Iy,
                            __global float* Ixt, __global float* Iyt, int w, int h)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float tx = x + u[i], ty = y + v[i];
    bool inside = tx >= 0.0f && ty >= 0.0f && tx <= w - 1.0f && ty <= h - 1.0f;
    float It = inside ? smp(B, w, h, tx, ty) - A[i] : 0.0f;
    Ixt[i] = Ix[i] * It;
    Iyt[i] = Iy[i] * It;
}

__kernel void solve(__global const float* A11, __global const float* A12, __global const float* A22,
                    __global const float* B1, __global const float* B2,
                    __global float* u, __global float* v, int w, int h, float reg)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float a = A11[i] + reg, c = A22[i] + reg, b = A12[i];
    float det = a * c - b * b;
    if (det <= 1e-12f) return;
    float b1 = B1[i], b2 = B2[i];
    float du = -(c * b1 - b * b2) / det;
    float dv = -(a * b2 - b * b1) / det;
    u[i] += clamp(du, -2.0f, 2.0f);
    v[i] += clamp(dv, -2.0f, 2.0f);
}

__kernel void median3(__global const float* src, __global float* dst, int w, int h)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float p[9];
    int k = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = clamp(y + dy, 0, h - 1);
        for (int dx = -1; dx <= 1; ++dx)
            p[k++] = src[yy * w + clamp(x + dx, 0, w - 1)];
    }
    for (int i = 1; i < 9; ++i) {           // Insertion Sort, 9 Werte
        float t = p[i]; int j = i - 1;
        while (j >= 0 && p[j] > t) { p[j + 1] = p[j]; --j; }
        p[j + 1] = t;
    }
    dst[y * w + x] = p[4];
}

__kernel void upsample_flow(__global const float* cu, __global const float* cv, int cw, int ch,
                            __global float* u, __global float* v, int w, int h)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float sx = (float)cw / w, sy = (float)ch / h;
    float cx = (x + 0.5f) * sx - 0.5f, cy = (y + 0.5f) * sy - 0.5f;
    u[y * w + x] = smp(cu, cw, ch, cx, cy) * (1.0f / sx);
    v[y * w + x] = smp(cv, cw, ch, cx, cy) * (1.0f / sy);
}

// g[i] = (x, y, dx, dy) in Koordinaten von I0; scale rechnet auf diese Stufe um.
__kernel void apply_guides(__global float* u, __global float* v, int w, int h,
                           __global const float4* g, int n, float scale, float k)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float S = 0, gu = 0, gv = 0;
    for (int i = 0; i < n; ++i) {
        float4 G = g[i];
        float dx = x - G.x * scale, dy = y - G.y * scale;
        float wgt = exp((dx * dx + dy * dy) * k);
        S += wgt; gu += wgt * G.z * scale; gv += wgt * G.w * scale;
    }
    if (S < 1e-4f) return;
    float nn = max(1.0f, S), keep = 1.0f - min(1.0f, S);
    int i = y * w + x;
    u[i] = u[i] * keep + gu / nn;
    v[i] = v[i] * keep + gv / nn;
}
)CLC",
// ---------------------------------------------------------------- Bewegungsunschaerfe
R"CLC(
// mode: 0 = vor+rueck, 1 = nur vorwaerts, 2 = nur rueckwaerts
__kernel void velocity(__global const float* fu, __global const float* fv,
                       __global const float* bu, __global const float* bv,
                       int fw, int fh, int mode, float scale, float halfShutter, float maxRadius, float minMotion,
                       __global float* VX, __global float* VY, int W, int H)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    float inv = 1.0f / scale;
    float fx = (x + 0.5f) * inv - 0.5f, fy = (y + 0.5f) * inv - 0.5f;
    float vx, vy;
    if (mode == 0) {
        vx = 0.5f * (smp(fu, fw, fh, fx, fy) - smp(bu, fw, fh, fx, fy));
        vy = 0.5f * (smp(fv, fw, fh, fx, fy) - smp(bv, fw, fh, fx, fy));
    } else if (mode == 1) {
        vx = smp(fu, fw, fh, fx, fy); vy = smp(fv, fw, fh, fx, fy);
    } else {
        vx = -smp(bu, fw, fh, fx, fy); vy = -smp(bv, fw, fh, fx, fy);
    }
    vx *= scale; vy *= scale;
    if (minMotion > 0.0f) {
        float l = sqrt(vx * vx + vy * vy);
        float k = l > minMotion ? (l - minMotion) / l : 0.0f;
        vx *= k; vy *= k;
    }
    vx *= halfShutter; vy *= halfShutter;
    float len = sqrt(vx * vx + vy * vy);
    if (len > maxRadius) { vx *= maxRadius / len; vy *= maxRadius / len; }
    VX[y * W + x] = vx; VY[y * W + x] = vy;
}

__kernel void row_max(__global const float* VX, __global const float* VY, int W, int H,
                      __global float* out)
{
    int y = get_global_id(0);
    if (y >= H) return;
    float m = 0;
    for (int x = 0; x < W; ++x) {
        float a = VX[y * W + x], b = VY[y * W + x];
        m = max(m, sqrt(a * a + b * b));
    }
    out[y] = m;
}

__kernel void tile_max(__global const float* VX, __global const float* VY, int W, int H,
                       int K, int TW, int TH, __global float* tmx, __global float* tmy)
{
    int tx = get_global_id(0), ty = get_global_id(1);
    if (tx >= TW || ty >= TH) return;
    float bx = 0, by = 0, bl = -1;
    for (int y = ty * K; y < min(H, ty * K + K); ++y)
        for (int x = tx * K; x < min(W, tx * K + K); ++x) {
            float vx = VX[y * W + x], vy = VY[y * W + x];
            float l = vx * vx + vy * vy;
            if (l > bl) { bl = l; bx = vx; by = vy; }
        }
    tmx[ty * TW + tx] = bx; tmy[ty * TW + tx] = by;
}

__kernel void neighbor_max(__global const float* tmx, __global const float* tmy, int TW, int TH,
                           __global float* nmx, __global float* nmy)
{
    int tx = get_global_id(0), ty = get_global_id(1);
    if (tx >= TW || ty >= TH) return;
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

// Natuerlicher Modus: Mittel entlang der eigenen Bewegungsbahn
__kernel void natural_blur(__global const float4* src, __global const float* VX, __global const float* VY,
                           int W, int H, int maxS, __global float4* dst)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    float vx = VX[y * W + x], vy = VY[y * W + x];
    float len = sqrt(vx * vx + vy * vy);
    if (len < 0.25f) { dst[y * W + x] = src[y * W + x]; return; }
    int S = min(maxS, (int)ceil(2.0f * len) + 1);
    if ((S & 1) == 0) ++S;
    float4 acc = (float4)(0.0f);
    for (int i = 0; i < S; ++i) {
        float t = (S == 1) ? 0.0f : -1.0f + 2.0f * i / (S - 1);
        acc += smp4(src, W, H, x + vx * t, y + vy * t);
    }
    dst[y * W + x] = acc * (1.0f / S);
}

inline float cone(float dist, float len) { return len > 1e-4f ? clamp(1.0f - dist / len, 0.0f, 1.0f) : 0.0f; }
inline float cylinder(float dist, float len)
{
    float a = 0.95f * len, b = 1.05f * len + 1e-4f;
    float t = clamp((dist - a) / (b - a), 0.0f, 1.0f);
    return 1.0f - t * t * (3.0f - 2.0f * t);
}

__kernel void reconstruct(__global const float4* src, __global const float* VX, __global const float* VY,
                          __global const float* nmx, __global const float* nmy,
                          int W, int H, int K, int TW, int maxS, float softZ,
                          __global float4* dst)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int ti = (y / K) * TW + (x / K);
    float nx = nmx[ti], ny = nmy[ti];
    float nLen = sqrt(nx * nx + ny * ny);
    float4 cX = src[y * W + x];
    if (nLen < 0.5f) { dst[y * W + x] = cX; return; }

    float vxX = VX[y * W + x], vyX = VY[y * W + x];
    float lenX = max(0.5f, sqrt(vxX * vxX + vyX * vyX));
    float wsum = 1.0f / lenX;
    float4 acc = cX * wsum;

    int S = min(maxS, max(3, (int)ceil(2.0f * nLen)));
    if ((S & 1) == 0) ++S;
    bool ownDir = sqrt(vxX * vxX + vyX * vyX) >= 0.5f;
    for (int i = 0; i < S; ++i) {
        float t = -1.0f + 2.0f * (i + 1.0f) / (S + 1.0f);
        bool useOwn = ownDir && (i & 1);
        float px = x + (useOwn ? vxX : nx) * t, py = y + (useOwn ? vyX : ny) * t;
        int ix = (int)floor(px + 0.5f), iy = (int)floor(py + 0.5f);
        if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;
        if (ix == x && iy == y) continue;
        float vxY = VX[iy * W + ix], vyY = VY[iy * W + ix];
        float lenY = max(0.5f, sqrt(vxY * vxY + vyY * vyY));
        float dist = sqrt((px - x) * (px - x) + (py - y) * (py - y));
        float f = clamp(1.0f - (lenX - lenY) / softZ, 0.0f, 1.0f);
        float b = clamp(1.0f - (lenY - lenX) / softZ, 0.0f, 1.0f);
        float a = f * cone(dist, lenY) + b * cone(dist, lenX)
                + 2.0f * cylinder(dist, lenY) * cylinder(dist, lenX);
        if (a <= 0.0f) continue;
        acc += smp4(src, W, H, px, py) * a;
        wsum += a;
    }
    dst[y * W + x] = acc / wsum;
}
)CLC"
};

}} // namespace fmb::gpu
