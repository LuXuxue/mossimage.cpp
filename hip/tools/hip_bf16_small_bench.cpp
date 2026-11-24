// Compare the small rocWMMA GEMM used by block/TE with the LDS-staged tiled GEMM.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bf16_gemm.h"
#include "hip_block.h"
#include "runtime.h"

using namespace mi::hip;
using bf16 = hip_block::bf16_bits;

static double time_it(int iters, auto&& fn) {
  fn();
  hipDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) fn();
  hipDeviceSynchronize();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
}

int main() {
  Device device(-1);
  struct Shape {
    int m, n, k;
  };
  const Shape shapes[] = {{64, 24576, 4096}, {64, 12288, 4096}, {64, 4096, 4096}, {64, 4096, 12288}};
  for (const Shape& s : shapes) {
    const std::size_t ab = static_cast<std::size_t>(s.m) * s.k, bb = static_cast<std::size_t>(s.n) * s.k;
    std::vector<bf16> ah(ab, 0x3f00), bh(bb, 0x3f00);
    DeviceBuffer da(ab * 2), db(bb * 2), dc(static_cast<std::size_t>(s.m) * s.n * 2);
    da.upload(ah.data(), ah.size() * 2);
    db.upload(bh.data(), bh.size() * 2);
    const double old_ms = time_it(4, [&] {
      hip_block::gemm(dc.as<bf16>(), s.n, da.as<bf16>(), s.k, db.as<bf16>(), s.k, s.m, s.n, s.k, 0);
    });
    const double tiled_ms = time_it(4, [&] {
      bf16_gemm_tiled(dc.as<bf16>(), s.n, da.as<bf16>(), s.k, db.as<bf16>(), s.k, s.m, s.n, s.k, 0);
    });
    std::printf("%4dx%-6dx%-5d rocWMMA-small %7.2f ms   tiled %7.2f ms   speedup %.2fx\n", s.m, s.n, s.k, old_ms,
                tiled_ms, old_ms / tiled_ms);
  }
  return 0;
}
