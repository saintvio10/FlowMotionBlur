// test_clapi.cpp - Prueft die selbst deklarierten OpenCL-Konstanten gegen die
// offiziellen Khronos-Header (nur bauen, wenn <CL/cl.h> vorhanden ist).
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "../src/gpu/ClApi.h"

namespace f = fmb::cl;
static_assert(f::kCL_SUCCESS == CL_SUCCESS, "CL_SUCCESS");
static_assert(f::kCL_TRUE == CL_TRUE, "CL_TRUE");
static_assert(f::kCL_DEVICE_TYPE_CPU == CL_DEVICE_TYPE_CPU, "CPU");
static_assert(f::kCL_DEVICE_TYPE_GPU == CL_DEVICE_TYPE_GPU, "GPU");
static_assert(f::kCL_DEVICE_TYPE_ALL == CL_DEVICE_TYPE_ALL, "ALL");
static_assert(f::kCL_MEM_READ_WRITE == CL_MEM_READ_WRITE, "RW");
static_assert(f::kCL_DEVICE_TYPE == CL_DEVICE_TYPE, "DEVICE_TYPE");
static_assert(f::kCL_DEVICE_GLOBAL_MEM_SIZE == CL_DEVICE_GLOBAL_MEM_SIZE, "MEM");
static_assert(f::kCL_DEVICE_NAME == CL_DEVICE_NAME, "NAME");
static_assert(f::kCL_PROGRAM_BUILD_LOG == CL_PROGRAM_BUILD_LOG, "LOG");
static_assert(f::kCL_CONTEXT_PLATFORM == CL_CONTEXT_PLATFORM, "PLATFORM");
static_assert(sizeof(f::cl_int) == sizeof(::cl_int) && sizeof(f::cl_ulong) == sizeof(::cl_ulong), "types");
static_assert(sizeof(f::cl_context_properties) == sizeof(::cl_context_properties), "props");

int main() { return 0; } // alle Pruefungen laufen beim Kompilieren
