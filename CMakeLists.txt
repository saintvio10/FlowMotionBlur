cmake_minimum_required(VERSION 3.20)
project(FlowMotionBlur CXX)
if(APPLE)
  enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional (nur Mac): Ordner mit den Premiere-GPU-Headern (PrSDKGPUFilter.h usw.,
# z. B. "Examples" eines neueren After Effects SDK). Damit wird Premieres eigener
# GPU-Weg (Metal) eingebaut.
set(PR_GPU_SDK "" CACHE PATH "Pfad zum Examples-Ordner mit Headers/PrSDKGPUFilter.h")

# Pfad zum "Examples"-Ordner des After Effects SDK, z. B.
#   -DAE_SDK="C:/AfterEffectsSDK/Examples"
set(AE_SDK "" CACHE PATH "Pfad zum Examples-Ordner des After Effects SDK")

find_package(Threads REQUIRED)

# --- Kern (ohne Adobe-SDK) -------------------------------------------------
add_library(fmb_core STATIC src/core/FlowCore.cpp src/gpu/GpuBackend.cpp)
target_include_directories(fmb_core PUBLIC src/core src/gpu)
target_link_libraries(fmb_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(fmb_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(fmb_core PRIVATE /O2 /fp:fast)
else()
  target_compile_options(fmb_core PRIVATE -O3)
endif()
# Intel-Macs: AVX2/FMA nutzen. Aktuelle Premiere-Versionen setzen auf Intel-Macs
# ohnehin mindestens einen Prozessor ab der 6. Generation (Skylake) voraus,
# der diese Befehle beherrscht. Der arm64-Teil (Apple-Chips) bleibt unberuehrt.
if(APPLE)
  set(FMB_X86_OPT "SHELL:-Xarch_x86_64 -mavx2" "SHELL:-Xarch_x86_64 -mfma")
  target_compile_options(fmb_core PRIVATE ${FMB_X86_OPT})
endif()

# --- Test (laeuft ueberall, auch ohne SDK) ---------------------------------
add_executable(fmb_test tests/test_core.cpp)
target_link_libraries(fmb_test PRIVATE fmb_core)
add_executable(fmb_test_occlusion tests/test_occlusion.cpp)
target_link_libraries(fmb_test_occlusion PRIVATE fmb_core)
add_executable(fmb_test_drone tests/test_drone.cpp)
add_executable(fmb_test_sequence tests/test_sequence.cpp)
target_link_libraries(fmb_test_sequence PRIVATE fmb_core)
target_link_libraries(fmb_test_drone PRIVATE fmb_core)

# Metal-Kerne fuer Premieres GPU-Weg (nur Mac) + Vergleichstest mit der CPU
if(APPLE)
  add_library(fmb_metal STATIC src/gpu/MetalBackend.mm)
  target_link_libraries(fmb_metal PUBLIC fmb_core "-framework Metal" "-framework Foundation")
  target_compile_options(fmb_metal PRIVATE -fobjc-arc -O3)
  set_target_properties(fmb_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_executable(fmb_test_metal tests/test_metal.mm)
  target_link_libraries(fmb_test_metal PRIVATE fmb_metal)
  target_compile_options(fmb_test_metal PRIVATE -fobjc-arc)
endif()

# Prueft die OpenCL-Deklarationen gegen die Khronos-Header (falls vorhanden)
find_path(OPENCL_HEADER_DIR CL/cl.h)
if(OPENCL_HEADER_DIR)
  add_executable(fmb_test_clapi tests/test_clapi.cpp)
  target_include_directories(fmb_test_clapi PRIVATE ${OPENCL_HEADER_DIR})
endif()

# --- Plugin -----------------------------------------------------------------
if(NOT AE_SDK)
  message(STATUS "AE_SDK nicht gesetzt - es wird nur der Kern + Test gebaut.")
  return()
endif()

set(AE_HEADERS "${AE_SDK}/Headers")
set(AE_INCLUDES "${AE_HEADERS}" "${AE_HEADERS}/SP" "${AE_SDK}/Util" "${AE_SDK}/Resources")
set(PIPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/FlowMotionBlurPiPL.r")

if(WIN32)
  # PiPL.r -> vorverarbeiten -> PiPLtool -> .rc -> in die DLL kompilieren
  set(PIPL_RR  "${CMAKE_CURRENT_BINARY_DIR}/FlowMotionBlurPiPL.rr")
  set(PIPL_RRC "${CMAKE_CURRENT_BINARY_DIR}/FlowMotionBlurPiPL.rrc")
  set(PIPL_RC  "${CMAKE_CURRENT_BINARY_DIR}/FlowMotionBlurPiPL.rc")
  add_custom_command(
    OUTPUT "${PIPL_RC}"
    COMMAND "${CMAKE_CXX_COMPILER}" /nologo /I "${AE_HEADERS}" /EP /P /Tc "${PIPL_SRC}" "/Fi${PIPL_RR}"
    COMMAND "${AE_SDK}/Resources/PiPLtool.exe" "${PIPL_RR}" "${PIPL_RRC}"
    COMMAND "${CMAKE_CXX_COMPILER}" /nologo /D MSWindows /EP /P /Tc "${PIPL_RRC}" "/Fi${PIPL_RC}"
    DEPENDS "${PIPL_SRC}"
    COMMENT "PiPL-Ressource erzeugen"
    VERBATIM)

  add_library(FlowMotionBlur MODULE src/plugin/FlowMotionBlur.cpp "${PIPL_RC}")
  target_include_directories(FlowMotionBlur PRIVATE ${AE_INCLUDES})
  target_compile_definitions(FlowMotionBlur PRIVATE MSWindows WIN32 _WINDOWS _CRT_SECURE_NO_WARNINGS)
  target_link_libraries(FlowMotionBlur PRIVATE fmb_core)
  set_target_properties(FlowMotionBlur PROPERTIES SUFFIX ".aex" PREFIX "")

elseif(APPLE)
  set(PIPL_RSRC "${CMAKE_CURRENT_BINARY_DIR}/FlowMotionBlur.rsrc")
  add_custom_command(
    OUTPUT "${PIPL_RSRC}"
    COMMAND xcrun Rez -o "${PIPL_RSRC}" -useDF -d AE_OS_MAC -d __MACH__
            -i "${AE_HEADERS}" -i "${AE_HEADERS}/SP" -i "${AE_SDK}/Resources"
            -arch x86_64 -arch arm64 "${PIPL_SRC}"
    DEPENDS "${PIPL_SRC}"
    COMMENT "PiPL-Ressource erzeugen"
    VERBATIM)

  set_source_files_properties("${PIPL_RSRC}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
  add_library(FlowMotionBlur MODULE src/plugin/FlowMotionBlur.cpp "${PIPL_RSRC}")
  target_include_directories(FlowMotionBlur PRIVATE ${AE_INCLUDES})
  target_link_libraries(FlowMotionBlur PRIVATE fmb_core "-framework CoreFoundation")
  set_target_properties(FlowMotionBlur PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION "plugin"
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/Info.plist.in")
  target_compile_options(FlowMotionBlur PRIVATE -Wno-multichar -O3 ${FMB_X86_OPT})

  # Premieres GPU-Weg (Metal): eigene Objektdatei mit den Premiere-Headern
  if(PR_GPU_SDK AND EXISTS "${PR_GPU_SDK}/Headers/PrSDKGPUFilter.h")
    add_library(fmb_prgpu OBJECT src/plugin/PrGpuFilter.cpp)
    target_include_directories(fmb_prgpu PRIVATE "${PR_GPU_SDK}/Headers" "${PR_GPU_SDK}/Headers/SP" src/core src/gpu)
    target_compile_options(fmb_prgpu PRIVATE -Wno-multichar -O3 ${FMB_X86_OPT})
    target_sources(FlowMotionBlur PRIVATE $<TARGET_OBJECTS:fmb_prgpu>)
    target_link_libraries(FlowMotionBlur PRIVATE fmb_metal)
    target_compile_definitions(FlowMotionBlur PRIVATE FMB_HAS_PR_GPU=1)
    message(STATUS "Premiere-GPU-Weg (Metal) wird eingebaut.")
  else()
    message(STATUS "PR_GPU_SDK nicht gesetzt - ohne Premiere-GPU-Weg.")
  endif()
endif()
