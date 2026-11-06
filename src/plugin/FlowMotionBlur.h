// FlowMotionBlur.h - Effekt-Definition fuer Premiere Pro / After Effects.

#pragma once

#define FMB_NAME          "Flow Motion Blur"
#define FMB_MATCH_NAME    "FMB FlowMotionBlur"
#define FMB_CATEGORY      "Flow Motion Blur"
#define FMB_DESCRIPTION   "Vektorbasierte Bewegungsunschaerfe per optischem Fluss."

#define FMB_MAJOR_VERSION 1
#define FMB_MINOR_VERSION 7
#define FMB_BUG_VERSION   0
#define FMB_STAGE_VERSION PF_Stage_DEVELOP
#define FMB_BUILD_VERSION 1
// Muss zu AE_Effect_Version in FlowMotionBlurPiPL.r passen:
// PF_VERSION(1,7,0,PF_Stage_DEVELOP,1) = (1<<19)|(7<<15)|1 = 753665

#define FMB_NUM_TRACK_POINTS 6

// Reihenfolge der Parameter (Index). Gruppen-Anfang/-Ende zaehlen mit.
enum {
    FMB_INPUT = 0,
    FMB_AMOUNT,          // Belichtungszeit (Shutter), 0.5 = 180 Grad
    FMB_MAX_SAMPLES,     // max. Samples pro Pixel
    FMB_MOTION_TYPE,     // Kamera-Modell (Drohne, Schwenk) oder frei
    FMB_BLUR_MODE,       // Natuerlich (Kamera) oder Objekte
    FMB_QUALITY,         // Aufloesung der Bewegungsanalyse
    FMB_SMOOTHING,       // Glaettung der Vektoren
    FMB_RGBA_TRACKING,   // Alphakanal fuer die Analyse mitverwenden
    FMB_USE_GPU,         // OpenCL verwenden, falls vorhanden
    FMB_VIEW,            // Ergebnis oder Vektoren anzeigen
    FMB_VECTOR_SOURCE,   // Automatisch (Tracking) oder Vektor-Ebene

    FMB_VEC_TOPIC_START, // --- Gruppe "Eigene Bewegungsvektoren"
    FMB_VEC_LAYER,       // Ebene mit Vektoren (R = X, G = Y)
    FMB_VEC_MAX_DISP,    // Pixel, die dem Wert 0 bzw. 1 entsprechen
    FMB_VEC_ENCODING,    // Kodierung
    FMB_VEC_FLIP_Y,      // Y-Richtung umkehren
    FMB_VEC_TOPIC_END,

    FMB_TP_TOPIC_START,  // --- Gruppe "Track Points"
    FMB_TP_RADIUS,       // Einflussradius in Pixeln
    FMB_TP_FIRST,        // danach je Punkt: [aktiv, Position]
    FMB_TP_TOPIC_END = FMB_TP_FIRST + 2 * FMB_NUM_TRACK_POINTS,

    FMB_NUM_PARAMS
};

#define FMB_TP_ON(i)  (FMB_TP_FIRST + 2 * (i))
#define FMB_TP_POS(i) (FMB_TP_FIRST + 2 * (i) + 1)

// Disk-IDs: duerfen sich nach Veroeffentlichung nie mehr aendern.
enum {
    AMOUNT_DISK_ID = 1,
    MAX_SAMPLES_DISK_ID,
    QUALITY_DISK_ID,
    SMOOTHING_DISK_ID,
    VIEW_DISK_ID,
    RGBA_TRACKING_DISK_ID,
    USE_GPU_DISK_ID,
    VECTOR_SOURCE_DISK_ID,
    VEC_TOPIC_START_DISK_ID,
    VEC_LAYER_DISK_ID,
    VEC_MAX_DISP_DISK_ID,
    VEC_ENCODING_DISK_ID,
    VEC_FLIP_Y_DISK_ID,
    VEC_TOPIC_END_DISK_ID,
    TP_TOPIC_START_DISK_ID,
    TP_RADIUS_DISK_ID,
    TP_TOPIC_END_DISK_ID,
    BLUR_MODE_DISK_ID,
    MOTION_TYPE_DISK_ID,
    TP_FIRST_DISK_ID = 100 // je Punkt i: 100 + 2i (aktiv), 101 + 2i (Position)
};

enum { QUALITY_DRAFT = 1, QUALITY_NORMAL, QUALITY_HIGH };
enum { MODE_NATURAL = 1, MODE_OBJECTS };
enum { MOTION_CAMERA = 1, MOTION_FREE };
enum { VIEW_RESULT = 1, VIEW_VECTORS, VIEW_DIAG };
enum { SOURCE_TRACKING = 1, SOURCE_LAYER };
enum { ENCODING_NORMALIZED = 1, ENCODING_PIXELS };
