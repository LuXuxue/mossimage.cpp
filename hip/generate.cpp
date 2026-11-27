// <chrono> before the HIP headers: their __noinline__ macro breaks libstdc++'s <format>, which <chrono> includes in C++20.
#include <chrono>

#include "generate.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "lora.h"
#include "text_encoder.h"
#include "vae.h"

namespace mi::hip {
namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / static_cast<double>(1ull << 30); }

// What the host can give to a pinned allocation: MemAvailable of /proc/meminfo on Linux (the page cache it would drop
// included), the physical memory the processes can use on Windows.
std::uint64_t host_available_bytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX s{};
  s.dwLength = sizeof s;
  if (!GlobalMemoryStatusEx(&s)) return std::numeric_limits<std::uint64_t>::max();
  return s.ullAvailPhys;
#else
  std::ifstream in("/proc/meminfo");
  std::string key;
  std::uint64_t kib = 0;
  while (in >> key >> kib) {
    if (key == "MemAvailable:") return kib * 1024;
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return std::numeric_limits<std::uint64_t>::max();  // unknown: not refused here
#endif
}

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("hip generate: " + what); }

}  // namespace

GenerateResult generate(Device& device, const GenerateRequest& r) {
  if (r.width == 0 || r.height == 0 || r.width % 16 != 0 || r.height % 16 != 0) fail("width and height must be multiples of 16");
  const std::uint32_t lh = r.height / 16, lw = r.width / 16;
  if (r.noise.size() != std::size_t{kDiTLatentChannels} * lh * lw) fail("noise must be [128, height / 16, width / 16]");
  if (r.timesteps.empty() || r.sigmas.size() != r.timesteps.size() + 1) fail("sigmas must have one more entry than timesteps");
  if (r.cond_ids.empty()) fail("the prompt has no tokens");
  const bool cfg = r.cfg != 1.0f;
  if (cfg && r.uncond_ids.empty()) fail("the negative prompt has no tokens");
  const auto report = [&](const char* stage, int done, int total) {
    if (r.progress) r.progress(stage, done, total);
  };

  GenerateResult out;
  out.width = r.width;
  out.height = r.height;
  // A component may be a single safetensors file (--dit/--te/--vae) instead of the bundle's directory.
  const auto component = [&](const std::filesystem::path& file, const char* sub) {
    return file.empty() ? r.model_dir / sub : file;
  };

  // The VAE first (about 0.1 GiB of weights, kept through the run). A full decode that fails is retried in tiles
  // after the text encoder and DiT have been freed.
  std::optional<VaeDecoder> vae;
  if (r.decode) {
    vae.emplace(device, component(r.vae_file, "vae/diffusion_pytorch_model.safetensors"));
    vae->set_profile(std::getenv("MI_HIP_VAE_PROFILE") != nullptr);
  }

  // The DiT's plan for the prompts' lengths: what it holds beside its weights, the steps' activations, the refiner's
  // scratch kept by the Dit (its input and SiTU-GLU buffers, at the longest prompt) and the refined text rows, plus a
  // reserve for the code objects and the runtime; and the device memory of its weights, what the device has free beside
  // that (the automatic budget) or what the VRAM budget leaves beside the VAE's weights and that, the smaller of the two,
  // less the allocations' rounding. The blocks past it stream from pinned host memory. A budget below
  // Dit::minimum_weight_bytes() is refused, never raised: the weights then keep within it. Checked from the token counts
  // before the text encoder runs, so a run that cannot be held is refused without it, and again from the encoder's
  // output lengths with the memory free when the DiT loads.
  constexpr std::uint64_t kBf16 = 2, kMiB = std::uint64_t{1} << 20, kReserve = 512 * kMiB;
  // hipMalloc rounds an allocation up to 2 MiB at worst: the weights' (the globals, the two refiner blocks, each block
  // on the device, the two slots) and the activations' (about 20 buffers).
  constexpr std::uint64_t kWeightRounding = (kDiTLayers + 5) * 2 * kMiB, kActivationMargin = 64 * kMiB;
  // On APUs and dGPUs with a large BAR, hipMalloc can back allocations from system shared memory after hipMemGetInfo
  // has reported little or no free memory. With the automatic budget we therefore let the allocator decide instead of
  // refusing on that report; an explicit --vram-budget remains a hard limit.
  constexpr std::uint64_t kAllocatorSlack = 16ull << 30;
  const std::uint64_t hw = std::uint64_t{lh} * lw;
  const std::uint64_t vae_bytes = vae ? vae->weight_bytes() : 0;
  char budget[64];
  if (r.vram_budget != 0) {
    std::snprintf(budget, sizeof budget, "a VRAM budget of %.2f GiB", gib(r.vram_budget));
  } else {
    std::snprintf(budget, sizeof budget, "the automatic budget");
  }
  struct DitPlan {
    std::uint64_t activations, weights;
  };
  const auto plan = [&](const std::vector<std::uint64_t>& lengths, std::uint64_t free) {
    std::uint64_t tokens = 0, text_rows = 0, longest = 0;
    for (const std::uint64_t length : lengths) {
      tokens += hw + length;
      text_rows += length;
      longest = std::max(longest, length);
    }
    const std::uint64_t activations = Dit::activation_bytes(tokens, lengths.size(), hw, r.int8_dit, !r.lora_file.empty()) +
                                      longest * (kDiTTextDim + 2 * std::uint64_t{kDiTFfnDim}) * kBf16 +
                                      text_rows * kDiTDim * kBf16;
    const std::uint64_t need = activations + kReserve;
    char msg[512];
    if (need > free && r.vram_budget == 0) free = need + kAllocatorSlack;
    if (need > free) {
      std::snprintf(msg, sizeof msg,
                    "the DiT at %ux%u needs %.2f GiB of device memory beside its weights (activations %.2f GiB, reserve "
                    "%.2f GiB); %s has %.2f GiB free",
                    r.width, r.height, gib(need), gib(activations), gib(kReserve), device.info().name.c_str(), gib(free));
      throw std::runtime_error(msg);
    }
    std::uint64_t room = free - need;
    if (r.vram_budget != 0) {
      if (r.vram_budget <= vae_bytes + need) {
        std::snprintf(msg, sizeof msg,
                      "%s leaves nothing for the DiT's weights: the VAE's weights take %.2f GiB and the DiT at %ux%u needs "
                      "%.2f GiB beside its weights (activations %.2f GiB, reserve %.2f GiB)",
                      budget, gib(vae_bytes), r.width, r.height, gib(need), gib(activations), gib(kReserve));
        throw std::runtime_error(msg);
      }
      room = std::min(room, r.vram_budget - vae_bytes - need);
    }
    const std::uint64_t weights = room > kWeightRounding ? room - kWeightRounding : 0;
    if (weights < Dit::minimum_weight_bytes()) {
      std::snprintf(msg, sizeof msg,
                    "%s leaves %.2f GiB of device memory for the DiT's weights beside the %.2f GiB it needs at %ux%u (%s "
                    "has %.2f GiB free); the weights take at least %.2f GiB, their embedders, modulations and text "
                    "refiner and two slots for the streamed blocks",
                    budget, gib(weights), gib(need), r.width, r.height, device.info().name.c_str(), gib(free),
                    gib(Dit::minimum_weight_bytes()));
      throw std::runtime_error(msg);
    }
    // The DiT's pinned host memory must be there to pin, with 1 GiB to spare for the rest of the process: the streamed
    // blocks, and the block-sized staging buffer the resident ones are uploaded through.
    const int resident = Dit::resident_blocks_for(weights);
    const std::uint64_t pinned = std::uint64_t(kDiTLayers - resident + (resident > 0 ? 1 : 0)) * Dit::block_bytes();
    constexpr std::uint64_t kHostSpare = std::uint64_t{1} << 30;
    if (const std::uint64_t available = host_available_bytes(); pinned + kHostSpare > available) {
      std::snprintf(msg, sizeof msg,
                    "%s keeps %d of the DiT's %d blocks on the device; loading it pins %.2f GiB of host memory (the other "
                    "%d blocks and a staging block) and leaves 1 GiB, this machine has %.2f GiB available",
                    budget, resident, kDiTLayers, gib(pinned), kDiTLayers - resident, gib(available));
      throw std::runtime_error(msg);
    }
    return DitPlan{activations, weights};
  };
  {
    std::vector<std::uint64_t> lengths{r.cond_ids.size()};
    if (cfg) lengths.push_back(r.uncond_ids.size());
    plan(lengths, device.free_memory());
  }

  // Text encoder: streamed layer by layer, freed before the DiT loads. The reset is explicit rather than relying on
  // the scope's destructor so the release happens right here, before the DiT's budget is measured; the synchronize
  // makes hipMemGetInfo see the reclaimed memory on drivers that defer the free until the queue drains.
  std::vector<std::vector<std::uint16_t>> embeds;
  {
    double t0 = now_s();
    report("text encoder load", 0, 1);
    TextEncoderOptions te_options;
    const char* ed = std::getenv("MI_HIP_TE_EXACT_DEQUANT");
    te_options.exact_dequant = ed != nullptr && std::strcmp(ed, "0") != 0;
    const char* pg = std::getenv("MI_HIP_TE_PACKED_GEMM");
    te_options.tiled_gemm = pg == nullptr || std::strcmp(pg, "0") == 0;
    const char* dd = std::getenv("MI_HIP_TE_DEVICE_DEQUANT");
    te_options.device_dequant = dd == nullptr || std::strcmp(dd, "0") != 0;
    std::optional<TextEncoder> encoder;
    encoder.emplace(device, component(r.te_file, "text_encoder"), te_options);
    const double load_s = now_s() - t0;
    out.text_encoder_s = load_s;
    report("text encoder load", 1, 1);
    t0 = now_s();
    report("text encoder", 0, 1);
    std::vector<std::vector<std::int32_t>> sequences{r.cond_ids};
    if (cfg) sequences.push_back(r.uncond_ids);
    embeds = encoder->encode(sequences);
    out.text_encoder_s = now_s() - t0;
    report("text encoder", 1, 1);
    out.text_encoder_load_s = load_s;
    encoder.reset();
    device.synchronize();
  }

  // DiT: resident, the text refiner once per prompt, then the steps.
  {
    double t0 = now_s();
    std::vector<std::uint64_t> lengths;
    for (const auto& e : embeds) lengths.push_back(e.size() / kDiTTextDim);
    const DitPlan p = plan(lengths, device.free_memory());
    DitOptions options;
    options.device_budget = p.weights;  // >= Dit::minimum_weight_bytes(), so never 0 (every block resident)
    std::optional<Lora> lora;
    if (!r.lora_file.empty()) {
      lora.emplace(r.lora_file, r.lora_scale, kDiTLayers);
      options.lora = &*lora;
    }
    Dit dit(device, component(r.dit_file, "transformer"), options);
    out.dit_resident_blocks = dit.resident_blocks();
    // What is left must hold the activations; the reserve absorbs the runtime's own allocations and the rounding.
    out.dit_free_bytes = device.free_memory();
    if (out.dit_free_bytes < p.activations + kActivationMargin) {
      char msg[512];
      std::snprintf(msg, sizeof msg,
                    "the DiT's weights take %.2f GiB of device memory (%s); beside them its activations at %ux%u need "
                    "%.2f GiB, %s reports %.2f GiB free",
                    gib(dit.weight_bytes()), budget, r.width, r.height, gib(p.activations + kActivationMargin),
                    device.info().name.c_str(), gib(out.dit_free_bytes));
      // With an explicit budget this is a real limit. With the automatic budget the report can be zero long before
      // shared memory is exhausted, so continue and let an actual hipMalloc failure report the real limit.
      if (r.vram_budget != 0) throw std::runtime_error(msg);
    }
    std::vector<DeviceBuffer> texts;
    texts.reserve(embeds.size());  // the rows point into it
    std::vector<Dit::Row> rows;
    for (std::size_t i = 0; i < embeds.size(); ++i) {
      const auto length = static_cast<std::uint32_t>(embeds[i].size() / kDiTTextDim);
      texts.push_back(dit.encode_text(embeds[i], length));
      rows.push_back({&texts.back(), length, i == 0 ? SkipSet{} : r.uncond_skip});
    }
    out.dit_load_s = now_s() - t0;

    t0 = now_s();
    std::vector<float> latents = r.noise;
    const int steps = static_cast<int>(r.timesteps.size());
    report("denoise", 0, steps);
    for (int s = 0; s < steps; ++s) {
      const auto si = static_cast<std::size_t>(s);
      const std::vector<std::vector<float>> pred = dit.forward(latents, lh, lw, r.timesteps[si], rows);
      // CFG then the official Euler step in the engine's op order (sampler_kernels.cu): u + g (c - u); x + h v, with
      // h = sigma_next - sigma rounded once from double. The fp32 expressions below compile without contraction.
      const float h = static_cast<float>(static_cast<double>(r.sigmas[si + 1]) - static_cast<double>(r.sigmas[si]));
      for (std::size_t i = 0; i < latents.size(); ++i) {
        float v = pred[0][i];
        if (cfg) {
          const float diff = pred[0][i] - pred[1][i];
          const float scaled = r.cfg * diff;
          v = pred[1][i] + scaled;
        }
        const float step = h * v;
        latents[i] = latents[i] + step;
      }
      report("denoise", s + 1, steps);
    }
    out.denoise_s = now_s() - t0;
    out.latents = std::move(latents);
  }
  device.synchronize();

  // VAE decode, after handing the final latents to the caller (a failed decode keeps them).
  if (r.on_latents) r.on_latents(out.latents);
  if (r.decode) {
    const double t0 = now_s();
    report("vae", 0, 1);
    VaeDecodeStats vst;
    // --vae-tiled is the only way to request tiling up front. A full decode that throws is retried here too; that is
    // an error fallback, not a memory prediction. Tiling defaults to the exact band schedule: it is byte-identical to
    // the full decode, so it has no tile-boundary brightness steps. Overlap-and-blend is opt-in because each tile
    // normalises by its own GroupNorm statistics and is measurably brighter or darker than its neighbours.
    const auto overlap_tiled = [&]() -> VaeDecodeStats {
      int tile_px = 512, overlap_px = 128;
      if (const char* tp = std::getenv("MI_HIP_VAE_TILE_PX")) tile_px = std::atoi(tp);
      if (const char* opx = std::getenv("MI_HIP_VAE_OVERLAP_PX")) overlap_px = std::atoi(opx);
      return vae->decode_tiled(out.latents.data(), lh, lw, out.rgb, nullptr,
                               static_cast<std::uint32_t>(std::max(tile_px, 32)),
                               static_cast<std::uint32_t>(std::max(overlap_px, 0)));
    };
    const auto tiled = [&]() -> VaeDecodeStats {
      if (std::getenv("MI_HIP_VAE_OVERLAP") != nullptr) return overlap_tiled();
      int tile = 512;
      if (const char* tp = std::getenv("MI_HIP_VAE_BAND_TILE_PX")) tile = std::atoi(tp);
      const VaeTilePlan plan = VaeDecoder::plan_tiled_min_memory(vae->base_channels(), lh, lw, tile);
      std::fprintf(stderr, "moss-infer-hip: exact band-tiled VAE: %s\n", plan.describe().c_str());
      try {
        return vae->decode_tiled_exact(out.latents.data(), lh, lw, out.rgb, plan);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "moss-infer-hip: band-tiled VAE failed (%s); falling back to overlap-and-blend\n", e.what());
        return overlap_tiled();
      }
    };
    if (r.vae_tiled) {
      vst = tiled();
    } else {
      try {
        vst = vae->decode(out.latents.data(), lh, lw, out.rgb);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "moss-infer-hip: full VAE decode failed (%s); decoding in tiles\n", e.what());
        vst = tiled();
      }
    }
    if (!vst.profile.empty()) {
      std::fprintf(stderr, "[vae-op]");
      for (const auto& [name, ms] : vst.profile) std::fprintf(stderr, " %s=%.0fms", name.c_str(), ms);
      std::fprintf(stderr, "\n");
    }
    out.vae_s = now_s() - t0;
    report("vae", 1, 1);
  }
  return out;
}

}  // namespace mi::hip
