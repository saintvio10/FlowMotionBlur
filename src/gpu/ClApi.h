// ClApi.h - Minimale OpenCL-1.2-Schnittstelle, zur Laufzeit geladen.
//
// Das Plugin linkt NICHT gegen OpenCL. Die Bibliothek wird beim ersten
// GPU-Aufruf gesucht (Windows: OpenCL.dll vom Grafiktreiber, macOS:
// OpenCL.framework). Fehlt sie, rechnet das Plugin einfach auf der CPU.
// Die Konstanten werden in tests/test_clapi.cpp gegen die offiziellen
// Khronos-Header geprueft.

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
    #define FMB_CL_CALL __stdcall
#else
    #define FMB_CL_CALL
#endif

namespace fmb { namespace cl {

typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint  cl_bool;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint  cl_device_info;
typedef cl_uint  cl_program_build_info;
typedef intptr_t cl_context_properties;

typedef struct _cl_platform_id*   cl_platform_id;
typedef struct _cl_device_id*     cl_device_id;
typedef struct _cl_context*       cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem*           cl_mem;
typedef struct _cl_program*       cl_program;
typedef struct _cl_kernel*        cl_kernel;
typedef struct _cl_event*         cl_event;

constexpr cl_int kCL_SUCCESS = 0;
constexpr cl_bool kCL_TRUE = 1;
constexpr cl_device_type kCL_DEVICE_TYPE_CPU = (1 << 1);
constexpr cl_device_type kCL_DEVICE_TYPE_GPU = (1 << 2);
constexpr cl_device_type kCL_DEVICE_TYPE_ALL = 0xFFFFFFFF;
constexpr cl_mem_flags kCL_MEM_READ_WRITE = (1 << 0);
constexpr cl_device_info kCL_DEVICE_TYPE            = 0x1000;
constexpr cl_device_info kCL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
constexpr cl_device_info kCL_DEVICE_NAME            = 0x102B;
constexpr cl_program_build_info kCL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_context_properties kCL_CONTEXT_PLATFORM = 0x1084;

struct Api {
    cl_int (FMB_CL_CALL *GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int (FMB_CL_CALL *GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    cl_int (FMB_CL_CALL *GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    cl_context (FMB_CL_CALL *CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                            void (FMB_CL_CALL *)(const char*, const void*, size_t, void*),
                                            void*, cl_int*);
    cl_command_queue (FMB_CL_CALL *CreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
    cl_program (FMB_CL_CALL *CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    cl_int (FMB_CL_CALL *BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
                                       void (FMB_CL_CALL *)(cl_program, void*), void*);
    cl_int (FMB_CL_CALL *GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
    cl_kernel (FMB_CL_CALL *CreateKernel)(cl_program, const char*, cl_int*);
    cl_int (FMB_CL_CALL *SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_mem (FMB_CL_CALL *CreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    cl_int (FMB_CL_CALL *EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*,
                                             cl_uint, const cl_event*, cl_event*);
    cl_int (FMB_CL_CALL *EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*,
                                            cl_uint, const cl_event*, cl_event*);
    cl_int (FMB_CL_CALL *EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                                               const size_t*, cl_uint, const cl_event*, cl_event*);
    cl_int (FMB_CL_CALL *Finish)(cl_command_queue);
    cl_int (FMB_CL_CALL *ReleaseMemObject)(cl_mem);
    cl_int (FMB_CL_CALL *ReleaseKernel)(cl_kernel);
    cl_int (FMB_CL_CALL *ReleaseProgram)(cl_program);
    cl_int (FMB_CL_CALL *ReleaseCommandQueue)(cl_command_queue);
    cl_int (FMB_CL_CALL *ReleaseContext)(cl_context);
};

// Laedt die OpenCL-Bibliothek. Gibt nullptr zurueck, wenn sie fehlt.
const Api* LoadApi();

}} // namespace fmb::cl
