// GpuBackend.cpp - OpenCL-Implementierung von Fluss und Bewegungsunschaerfe.

#include "GpuBackend.h"
#include "ClApi.h"
#include "Kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace fmb {
namespace cl {

// ---------------------------------------------------------------------------
// Bibliothek laden
// ---------------------------------------------------------------------------

static void* OpenLibrary()
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA("OpenCL.dll"));
#elif defined(__APPLE__)
    return dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_NOW | RTLD_LOCAL);
#else
    void* h = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    return h ? h : dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
#endif
}

static void* Symbol(void* lib, const char* name)
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

const Api* LoadApi()
{
    static Api api;
    static bool ok = false;
    static std::once_flag once;
    std::call_once(once, [] {
        void* lib = OpenLibrary();
        if (!lib) return;
        bool all = true;
        auto load = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(Symbol(lib, name));
            all = all && fn != nullptr;
        };
        load(api.GetPlatformIDs, "clGetPlatformIDs");
        load(api.GetDeviceIDs, "clGetDeviceIDs");
        load(api.GetDeviceInfo, "clGetDeviceInfo");
        load(api.CreateContext, "clCreateContext");
        load(api.CreateCommandQueue, "clCreateCommandQueue");
        load(api.CreateProgramWithSource, "clCreateProgramWithSource");
        load(api.BuildProgram, "clBuildProgram");
        load(api.GetProgramBuildInfo, "clGetProgramBuildInfo");
        load(api.CreateKernel, "clCreateKernel");
        load(api.SetKernelArg, "clSetKernelArg");
        load(api.CreateBuffer, "clCreateBuffer");
        load(api.EnqueueWriteBuffer, "clEnqueueWriteBuffer");
        load(api.EnqueueReadBuffer, "clEnqueueReadBuffer");
        load(api.EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
        load(api.Finish, "clFinish");
        load(api.ReleaseMemObject, "clReleaseMemObject");
        load(api.ReleaseKernel, "clReleaseKernel");
        load(api.ReleaseProgram, "clReleaseProgram");
        load(api.ReleaseCommandQueue, "clReleaseCommandQueue");
        load(api.ReleaseContext, "clReleaseContext");
        ok = all;
    });
    return ok ? &api : nullptr;
}

} // namespace cl

namespace gpu {

using namespace fmb::cl;

// ---------------------------------------------------------------------------
// Geraet und Programm (einmalig)
// ---------------------------------------------------------------------------

struct State {
    const Api*       api = nullptr;
    cl_context       ctx = nullptr;
    cl_device_id     dev = nullptr;
    cl_command_queue queue = nullptr;
    cl_program       prog = nullptr;
    std::unordered_map<std::string, cl_kernel> kernels;
    std::string      deviceName;
    std::mutex       mtx; // eine GPU-Pipeline zur Zeit
};

static bool EnvSet(const char* name)
{
    const char* v = std::getenv(name);
    return v && *v && *v != '0';
}

static State* CreateState()
{
    if (EnvSet("FMB_DISABLE_GPU")) return nullptr;
    const Api* api = LoadApi();
    if (!api) return nullptr;

    const bool allowCpu = EnvSet("FMB_OPENCL_ALLOW_CPU");

    cl_platform_id platforms[16];
    cl_uint np = 0;
    if (api->GetPlatformIDs(16, platforms, &np) != kCL_SUCCESS || np == 0) return nullptr;

    // Geraet mit dem meisten Speicher waehlen, GPUs bevorzugt.
    cl_device_id best = nullptr; cl_platform_id bestPlat = nullptr;
    cl_ulong bestMem = 0; bool bestIsGpu = false;
    for (cl_uint p = 0; p < std::min<cl_uint>(np, 16); ++p) {
        cl_device_id devs[16]; cl_uint nd = 0;
        if (api->GetDeviceIDs(platforms[p], kCL_DEVICE_TYPE_ALL, 16, devs, &nd) != kCL_SUCCESS) continue;
        for (cl_uint d = 0; d < std::min<cl_uint>(nd, 16); ++d) {
            cl_device_type type = 0; cl_ulong mem = 0;
            api->GetDeviceInfo(devs[d], kCL_DEVICE_TYPE, sizeof(type), &type, nullptr);
            api->GetDeviceInfo(devs[d], kCL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
            const bool isGpu = (type & kCL_DEVICE_TYPE_GPU) != 0;
            if (!isGpu && !allowCpu) continue;
            if ((isGpu && !bestIsGpu) || (isGpu == bestIsGpu && mem > bestMem)) {
                best = devs[d]; bestPlat = platforms[p]; bestMem = mem; bestIsGpu = isGpu;
            }
        }
    }
    if (!best) return nullptr;

    std::unique_ptr<State> s(new State);
    s->api = api;
    s->dev = best;

    char name[256] = {0};
    api->GetDeviceInfo(best, kCL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
    s->deviceName = name;

    cl_int err = kCL_SUCCESS;
    cl_context_properties props[] = { kCL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(bestPlat), 0 };
    s->ctx = api->CreateContext(props, 1, &best, nullptr, nullptr, &err);
    if (err != kCL_SUCCESS || !s->ctx) return nullptr;
    s->queue = api->CreateCommandQueue(s->ctx, best, 0, &err);
    if (err != kCL_SUCCESS || !s->queue) return nullptr;

    const cl_uint nParts = sizeof(kKernelSourceParts) / sizeof(kKernelSourceParts[0]);
    s->prog = api->CreateProgramWithSource(s->ctx, nParts, const_cast<const char**>(kKernelSourceParts), nullptr, &err);
    if (err != kCL_SUCCESS || !s->prog) return nullptr;
    if (api->BuildProgram(s->prog, 1, &best, "", nullptr, nullptr) != kCL_SUCCESS) {
        if (EnvSet("FMB_GPU_DEBUG")) {
            std::vector<char> log(65536, 0);
            api->GetProgramBuildInfo(s->prog, best, kCL_PROGRAM_BUILD_LOG, log.size() - 1, log.data(), nullptr);
            std::fprintf(stderr, "[FlowMotionBlur] OpenCL-Build fehlgeschlagen:\n%s\n", log.data());
        }
        return nullptr;
    }

    const char* names[] = { "downscale", "gradients", "box_h", "box_v", "warp_products", "solve",
                            "median3", "upsample_flow", "apply_guides", "velocity", "row_max",
                            "tile_max", "neighbor_max", "reconstruct", "natural_blur" };
    for (const char* n : names) {
        cl_kernel k = api->CreateKernel(s->prog, n, &err);
        if (err != kCL_SUCCESS || !k) return nullptr;
        s->kernels[n] = k;
    }
    return s.release(); // lebt bis Programmende
}

static State* GetState()
{
    static State* state = nullptr;
    static std::once_flag once;
    std::call_once(once, [] { state = CreateState(); });
    return state;
}

bool Available() { return GetState() != nullptr; }

std::string DeviceName()
{
    State* s = GetState();
    return s ? s->deviceName : std::string();
}

// ---------------------------------------------------------------------------
// Sitzung: Puffer + Kernel-Aufrufe, merkt sich den ersten Fehler
// ---------------------------------------------------------------------------

struct Buf {
    const Api* api = nullptr;
    cl_mem     mem = nullptr;
    Buf() = default;
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : api(o.api), mem(o.mem) { o.mem = nullptr; }
    Buf& operator=(Buf&& o) noexcept { reset(); api = o.api; mem = o.mem; o.mem = nullptr; return *this; }
    ~Buf() { reset(); }
    void reset() { if (mem) api->ReleaseMemObject(mem); mem = nullptr; }
};

class Session {
public:
    explicit Session(State* s) : s_(s), api_(s->api) {}

    bool ok() const { return ok_; }

    Buf Alloc(size_t floats)
    {
        Buf b; b.api = api_;
        if (!ok_) return b;
        cl_int err = kCL_SUCCESS;
        b.mem = api_->CreateBuffer(s_->ctx, kCL_MEM_READ_WRITE, std::max<size_t>(1, floats) * sizeof(float), nullptr, &err);
        if (err != kCL_SUCCESS || !b.mem) ok_ = false;
        return b;
    }

    Buf Upload(const float* data, size_t floats)
    {
        Buf b = Alloc(floats);
        if (ok_ && floats > 0)
            Check(api_->EnqueueWriteBuffer(s_->queue, b.mem, kCL_TRUE, 0, floats * sizeof(float), data, 0, nullptr, nullptr));
        return b;
    }
    Buf Upload(const std::vector<float>& v) { return Upload(v.data(), v.size()); }

    void Download(const Buf& b, float* data, size_t floats)
    {
        if (ok_ && floats > 0)
            Check(api_->EnqueueReadBuffer(s_->queue, b.mem, kCL_TRUE, 0, floats * sizeof(float), data, 0, nullptr, nullptr));
    }

    template <class... Args>
    void Run(const char* kernel, size_t gx, size_t gy, const Args&... args)
    {
        if (!ok_ || gx == 0 || gy == 0) return;
        cl_kernel k = s_->kernels[kernel];
        cl_uint i = 0;
        (SetArg(k, i++, Conv(args)), ...);
        size_t global[2] = { gx, gy };
        if (ok_) Check(api_->EnqueueNDRangeKernel(s_->queue, k, gy > 1 ? 2 : 1, nullptr, global, nullptr, 0, nullptr, nullptr));
    }

    // Separabler Box-Filter (r = 0 kopiert).
    Buf Box(const Buf& src, int w, int h, int r)
    {
        const size_t n = static_cast<size_t>(w) * h;
        Buf tmp = Alloc(n), dst = Alloc(n);
        Run("box_h", w, h, src, tmp, w, h, r);
        Run("box_v", w, h, tmp, dst, w, h, r);
        return dst;
    }

    Buf Zeros(size_t floats) { std::vector<float> z(floats, 0.f); return Upload(z); }

private:
    static cl_mem Conv(const Buf& b) { return b.mem; }
    static cl_int Conv(int v)        { return v; }
    static float  Conv(float v)      { return v; }

    template <class T>
    void SetArg(cl_kernel k, cl_uint i, const T& v)
    {
        if (ok_) Check(api_->SetKernelArg(k, i, sizeof(T), &v));
    }
    void Check(cl_int e) { if (e != kCL_SUCCESS) ok_ = false; }

    State*     s_;
    const Api* api_;
    bool       ok_ = true;
};

// ---------------------------------------------------------------------------
// Optischer Fluss
// ---------------------------------------------------------------------------

static float GuideK(float radius, float scale)
{
    float sigma = std::max(1.f, radius * scale);
    return -1.f / (2.f * sigma * sigma);
}

bool ComputeFlow(const Image& I0, const Image& I1, const FlowParams& p, Flow& out,
                 const std::vector<Guide>* guides)
{
    State* st = GetState();
    if (!st || I0.empty() || I0.w != I1.w || I0.h != I1.h) return false;
    std::lock_guard<std::mutex> lock(st->mtx);
    Session S(st);

    // Pyramidengroessen wie auf der CPU
    std::vector<int> W(1, I0.w), H(1, I0.h);
    while (static_cast<int>(W.size()) < std::max(1, p.maxLevels)) {
        if (std::min(W.back(), H.back()) / 2 < 16) break;
        W.push_back((W.back() + 1) / 2);
        H.push_back((H.back() + 1) / 2);
    }
    const int levels = static_cast<int>(W.size());

    std::vector<Buf> P0, P1;
    P0.push_back(S.Upload(I0.d));
    P1.push_back(S.Upload(I1.d));
    for (int l = 1; l < levels; ++l) {
        const size_t n = static_cast<size_t>(W[l]) * H[l];
        P0.push_back(S.Alloc(n));
        P1.push_back(S.Alloc(n));
        S.Run("downscale", W[l], H[l], P0[l - 1], W[l - 1], H[l - 1], P0[l], W[l], H[l], 2);
        S.Run("downscale", W[l], H[l], P1[l - 1], W[l - 1], H[l - 1], P1[l], W[l], H[l], 2);
    }

    Buf gbuf;
    int ng = 0;
    if (guides && !guides->empty()) {
        std::vector<float> g;
        for (const Guide& q : *guides) { g.push_back(q.x); g.push_back(q.y); g.push_back(q.dx); g.push_back(q.dy); }
        gbuf = S.Upload(g);
        ng = static_cast<int>(guides->size());
    }

    Buf u, v;
    int cw = 0, ch = 0;
    for (int L = levels - 1; L >= 0; --L) {
        const int w = W[L], h = H[L];
        const size_t n = static_cast<size_t>(w) * h;

        Buf nu, nv;
        if (L == levels - 1) {
            const float ls = static_cast<float>(w) / I0.w;
            Flow init; CoarseInit(p, I0.w, w, h, init); // Startfeld bzw. globale Verschiebung
            nu = S.Upload(init.u.d);
            nv = S.Upload(init.v.d);
            if (ng > 0) {
                float scale = static_cast<float>(w) / I0.w;
                S.Run("apply_guides", w, h, nu, nv, w, h, gbuf, ng, scale, GuideK(p.guideRadius, scale));
            }
        } else {
            nu = S.Alloc(n); nv = S.Alloc(n);
            S.Run("upsample_flow", w, h, u, v, cw, ch, nu, nv, w, h);
        }
        u = std::move(nu); v = std::move(nv);
        cw = w; ch = h;

        Buf Ix = S.Alloc(n), Iy = S.Alloc(n), Ixx = S.Alloc(n), Ixy = S.Alloc(n), Iyy = S.Alloc(n);
        S.Run("gradients", w, h, P0[L], w, h, Ix, Iy, Ixx, Ixy, Iyy);
        Buf A11 = S.Box(Ixx, w, h, p.windowRadius);
        Buf A12 = S.Box(Ixy, w, h, p.windowRadius);
        Buf A22 = S.Box(Iyy, w, h, p.windowRadius);

        Buf Ixt = S.Alloc(n), Iyt = S.Alloc(n);
        for (int it = 0; it < std::max(1, p.iterations); ++it) {
            S.Run("warp_products", w, h, P0[L], P1[L], u, v, Ix, Iy, Ixt, Iyt, w, h);
            Buf B1 = S.Box(Ixt, w, h, p.windowRadius);
            Buf B2 = S.Box(Iyt, w, h, p.windowRadius);
            S.Run("solve", w, h, A11, A12, A22, B1, B2, u, v, w, h, p.regularization);
        }

        Buf mu = S.Alloc(n), mv = S.Alloc(n);
        S.Run("median3", w, h, u, mu, w, h);
        S.Run("median3", w, h, v, mv, w, h);
        u = std::move(mu); v = std::move(mv);
    }

    const int w = I0.w, h = I0.h;
    Buf fu = S.Box(u, w, h, std::max(0, p.smoothRadius));
    Buf fv = S.Box(v, w, h, std::max(0, p.smoothRadius));
    if (ng > 0) S.Run("apply_guides", w, h, fu, fv, w, h, gbuf, ng, 1.f, GuideK(p.guideRadius, 1.f));

    Flow res;
    res.u.resize(w, h);
    res.v.resize(w, h);
    S.Download(fu, res.u.d.data(), res.u.d.size());
    S.Download(fv, res.v.d.data(), res.v.d.size());
    if (!S.ok()) return false;
    out = std::move(res);
    return true;
}

// ---------------------------------------------------------------------------
// Bewegungsunschaerfe
// ---------------------------------------------------------------------------

bool ApplyMotionBlur(const RGBAImage& src, const Flow* fwd, const Flow* bwd,
                     int flowScale, const BlurParams& p, RGBAImage& dst)
{
    if (fwd && fwd->empty()) fwd = nullptr;
    if (bwd && bwd->empty()) bwd = nullptr;
    const int W = src.w, H = src.h;
    if ((!fwd && !bwd) || p.shutter <= 0.f || W <= 0 || H <= 0) {
        dst.resize(W, H);
        dst.d = src.d;
        return true;
    }
    if (fwd && bwd && (fwd->w() != bwd->w() || fwd->h() != bwd->h())) return false;

    State* st = GetState();
    if (!st) return false;
    std::lock_guard<std::mutex> lock(st->mtx);
    Session S(st);

    const Flow& any = fwd ? *fwd : *bwd;
    const int fw = any.w(), fh = any.h();
    const int mode = (fwd && bwd) ? 0 : (fwd ? 1 : 2);

    Buf img = S.Upload(src.d);
    Buf fu = fwd ? S.Upload(fwd->u.d) : S.Alloc(1);
    Buf fv = fwd ? S.Upload(fwd->v.d) : S.Alloc(1);
    Buf bu = bwd ? S.Upload(bwd->u.d) : S.Alloc(1);
    Buf bv = bwd ? S.Upload(bwd->v.d) : S.Alloc(1);

    const size_t n = static_cast<size_t>(W) * H;
    Buf VX = S.Alloc(n), VY = S.Alloc(n);
    const float scale = static_cast<float>(std::max(1, flowScale));
    S.Run("velocity", W, H, fu, fv, bu, bv, fw, fh, mode, scale, 0.5f * p.shutter, kMaxBlurRadius, p.minMotion, VX, VY, W, H);

    Buf rm = S.Alloc(H);
    S.Run("row_max", H, 1, VX, VY, W, H, rm);
    std::vector<float> rowMax(H, 0.f);
    S.Download(rm, rowMax.data(), rowMax.size());
    if (!S.ok()) return false;
    float globalMax = 0.f;
    for (float m : rowMax) globalMax = std::max(globalMax, m);

    dst.ensure(W, H); // wird vollstaendig ueberschrieben
    if (globalMax < 0.5f) { dst.d = src.d; return true; }

    if (p.mode == kBlurNatural) {
        Buf out = S.Alloc(n * 4);
        S.Run("natural_blur", W, H, img, VX, VY, W, H, std::max(1, p.maxSamples), out);
        S.Download(out, dst.d.data(), dst.d.size());
        return S.ok();
    }

    const int K = std::max(4, static_cast<int>(std::ceil(globalMax)));
    const int TW = (W + K - 1) / K, TH = (H + K - 1) / K;
    const size_t nt = static_cast<size_t>(TW) * TH;
    Buf tmx = S.Alloc(nt), tmy = S.Alloc(nt), nmx = S.Alloc(nt), nmy = S.Alloc(nt);
    S.Run("tile_max", TW, TH, VX, VY, W, H, K, TW, TH, tmx, tmy);
    S.Run("neighbor_max", TW, TH, tmx, tmy, TW, TH, nmx, nmy);

    Buf out = S.Alloc(n * 4);
    S.Run("reconstruct", W, H, img, VX, VY, nmx, nmy, W, H, K, TW, std::max(3, p.maxSamples), kSoftZ, out);
    S.Download(out, dst.d.data(), dst.d.size());
    return S.ok();
}

}} // namespace fmb::gpu
