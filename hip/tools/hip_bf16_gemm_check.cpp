// Check the tiled bf16 GEMM against a host replay on LoRA-like and tail shapes.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bf16_gemm.h"
#include "half.h"
#include "runtime.h"

using namespace mi::hip;
using bf16 = hip_block::bf16_bits;

static bf16 rbf(float v) { return mi::hip::float_to_bf16(v); }

int main() {
  Device device(-1);
  std::mt19937 gen(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int cases[][3] = {{37, 80, 64}, {65, 96, 128}, {129, 160, 128}, {256, 512, 128}, {521, 128, 3072}};
  bool ok = true;
  for (const auto& c : cases) {
    const int m = c[0], n = c[1], k = c[2];
    std::vector<bf16> ah(static_cast<std::size_t>(m) * k), bh(static_cast<std::size_t>(n) * k);
    for (auto& v : ah) v = rbf(nd(gen));
    for (auto& v : bh) v = rbf(nd(gen));
    DeviceBuffer da(ah.size() * 2), db(bh.size() * 2), dc(static_cast<std::size_t>(m) * n * 2);
    da.upload(ah.data(), ah.size() * 2);
    db.upload(bh.data(), bh.size() * 2);
    bf16_gemm_tiled(dc.as<bf16>(), n, da.as<bf16>(), k, db.as<bf16>(), k, m, n, k, 0);
    device.synchronize();
    std::vector<bf16> got(static_cast<std::size_t>(m) * n);
    dc.download(got.data(), got.size() * 2);
    std::size_t bad = 0;
    for (int r = 0; r < m; ++r) {
      for (int col = 0; col < n; ++col) {
        float acc = 0.0f;
        for (int t = 0; t < k; ++t) {
          const float av = mi::hip::bf16_to_float(ah[static_cast<std::size_t>(r) * k + t]);
          const float bv = mi::hip::bf16_to_float(bh[static_cast<std::size_t>(col) * k + t]);
          acc += av * bv;
        }
        if (got[static_cast<std::size_t>(r) * n + col] != rbf(acc)) ++bad;
      }
    }
    // Compare against the original small-tile GEMM on the same operands.
    DeviceBuffer dold(static_cast<std::size_t>(m) * n * 2);
    hip_block::gemm(dold.as<bf16>(), n, da.as<bf16>(), k, db.as<bf16>(), k, m, n, k, 0);
    device.synchronize();
    std::vector<bf16> old(static_cast<std::size_t>(m) * n);
    dold.download(old.data(), old.size() * 2);
    std::size_t diff = 0, big = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (got[i] == old[i]) continue;
      ++diff;
      const float a = mi::hip::bf16_to_float(got[i]), b = mi::hip::bf16_to_float(old[i]);
      if (std::fabs(a - b) > 0.01f * (1.0f + std::fabs(b))) ++big;
    }
    std::printf("%4dx%4dx%4d: %s (%zu of %d) vs-old diff=%zu big=%zu\n", m, n, k, bad ? "FAILED" : "ok", bad,
                m * n, diff, big);
    ok &= bad == 0;
  }
  return ok ? 0 : 1;
}
