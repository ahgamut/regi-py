#ifndef FEATURIZE_H
#define FEATURIZE_H
#include <card.h>
#include <combo.h>
#include <phaseinfo.h>
#include <vector>

/* Shared, architecture-independent featurization kernels -- the C++ home of the
 * numpy glue in rl/features.py, so training (pybind) and the browser (Embind) call
 * ONE featurizer. Binding-agnostic (no pybind/emscripten here). The heavy work is
 * already C++ (LocationInfo/ComboTable/PhaseInfo::comboDamage/comboBlock); these
 * assemble the float32 arrays the nets consume. Bit-for-bit parity with the old
 * numpy is mandatory -- watch the float32-vs-float64 divide per function:
 *   capabilities / location L1-normalize: float32 divide (numpy array ops)
 *   candidate features:                    float64 divide then cast (python scalars) */
namespace regi
{
    constexpr i32 CAP_CHANNELS = 2;       // per-card [attack, defense]
    constexpr float CAP_SCALE = 40.0f;    // King: 40 HP -> -1.0, 20 dmg -> -0.5
    constexpr i32 CAND_FEATURE_DIM = 9;   // CAND_FEATURE_NAMES in rl/features.py
    // one frame's per-card token width: location(9) + used_pile(22) + capability(2)
    constexpr i32 FEATURE_WIDTH = 33;

    // (MAX_CARDS_IN_GAME * CAP_CHANNELS) row-major, scaled 1/CAP_SCALE (float32 math).
    std::vector<float> cardCapabilities(const PhaseInfo &phase);
    // (MAX_CARDS_IN_GAME * MAX_LOCATIONS) row-major, L1 row-normalized (float32 math).
    std::vector<float> locationArray(const PhaseInfo &phase, i32 perspective);
    // (MAX_CARDS_IN_GAME * MAX_PLAYED_STATUS) row-major, u32 -> float32.
    std::vector<float> usedPileArray(const PhaseInfo &phase);
    // (K * CAND_FEATURE_DIM) row-major candidate features (float64 divide -> float32).
    std::vector<float> candidateSemantics(const PhaseInfo &phase,
                                          const std::vector<Combo> &combos);
    // (MAX_CARDS_IN_GAME * (window * FEATURE_WIDTH)) frame-major fused card tokens over
    // `phases` (oldest -> newest; window = phases.size()); the card-token net layout.
    std::vector<float> fuseCardTokens(const std::vector<PhaseInfo> &phases, i32 perspective);
}
#endif
