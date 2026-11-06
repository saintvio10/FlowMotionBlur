// FlowMotionBlurPiPL.r - Plugin-Beschreibung (PiPL), die Premiere Pro und
// After Effects beim Start lesen. Werte muessen zu GlobalSetup() passen.

#include "AEConfig.h"
#include "AE_EffectVers.h"

#ifndef AE_OS_WIN
    #include <AE_General.r>
#endif

resource 'PiPL' (16000) {
    {
        Kind {
            AEEffect
        },
        Name {
            "Flow Motion Blur"
        },
        Category {
            "Flow Motion Blur"
        },
#ifdef AE_OS_WIN
    #if defined(AE_PROC_ARM64)
        CodeWinARM64 {"EffectMain"},
    #else
        CodeWin64X86 {"EffectMain"},
    #endif
#elif defined(AE_OS_MAC)
        CodeMacIntel64 {"EffectMain"},
        CodeMacARM64 {"EffectMain"},
#endif
        AE_PiPL_Version {
            2,
            0
        },
        AE_Effect_Spec_Version {
            PF_PLUG_IN_VERSION,
            PF_PLUG_IN_SUBVERS
        },
        AE_Effect_Version {
            786433      /* PF_VERSION(1,8,0,PF_Stage_DEVELOP,1) */
        },
        AE_Effect_Info_Flags {
            0
        },
        AE_Effect_Global_OutFlags {
            0x02000006  /* WIDE_TIME_INPUT | NON_PARAM_VARY | DEEP_COLOR_AWARE */
        },
        AE_Effect_Global_OutFlags_2 {
            0x00000000
        },
        AE_Effect_Match_Name {
            "FMB FlowMotionBlur"
        },
        AE_Reserved_Info {
            0
        }
    }
};
