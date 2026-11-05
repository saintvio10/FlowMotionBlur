// GpuBackend.h - GPU-Beschleunigung per OpenCL.
//
// Gleiche Schnittstelle wie der CPU-Kern. Jede Funktion gibt false zurueck,
// wenn keine GPU verfuegbar ist oder ein Fehler auftritt; der Aufrufer
// rechnet dann auf der CPU weiter.
//
// Umgebungsvariablen:
//   FMB_DISABLE_GPU=1         GPU nie verwenden
//   FMB_OPENCL_ALLOW_CPU=1    auch OpenCL-CPU-Geraete zulassen (zum Testen)

#pragma once

#include "../core/FlowCore.h"

#include <string>

namespace fmb { namespace gpu {

// true, wenn ein OpenCL-Geraet bereitsteht (initialisiert beim ersten Aufruf).
bool Available();

// Name des verwendeten Geraets (leer, wenn keins).
std::string DeviceName();

bool ComputeFlow(const Image& I0, const Image& I1, const FlowParams& p, Flow& out,
                 const std::vector<Guide>* guides = nullptr);

bool ApplyMotionBlur(const RGBAImage& src, const Flow* fwd, const Flow* bwd,
                     int flowScale, const BlurParams& p, RGBAImage& dst);

}} // namespace fmb::gpu
