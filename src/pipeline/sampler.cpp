#include "pipeline/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"

namespace mi::pipeline {
namespace {

void check_latent(const Tensor& t, const char* what) {
  if (t.device != Device::kCUDA || t.dtype != DType::kF32 || !t.is_contiguous() || t.rank != 3 ||
      t.dim(0) != moss_dit::kLatentChannels) {
    throw std::invalid_argument(std::string("pipeline::denoise: ") + what + " must be contiguous CUDA fp32 [128, h, w], got " +
                                t.describe());
  }
}

}  // namespace

bool Guidance::skip_active(int step, int num_steps) const {
  // P:409: Python float division, compared as doubles.
  const double pos = static_cast<double>(step) / static_cast<double>(num_steps);
  return slg_start <= pos && pos < slg_stop;
}

bool Guidance::slg_active(int step, int num_steps) const {
  return slg_layers.any() && slg_scale > 0 && skip_active(step, num_steps);
}

double Guidance::vision_weight_at(int step, int num_steps) const {
  if (!vision_plain) return 1.0;
  const double pos = static_cast<double>(step) / static_cast<double>(num_steps);
  return vision_start <= pos && pos < vision_stop ? vision_weight : 0.0;
}

bool Guidance::cfg_active(int step, int num_steps) const {
  return do_cfg() && (cfg_truncation >= 1.0 || static_cast<double>(step) / static_cast<double>(num_steps) < cfg_truncation);
}

std::vector<std::int64_t> denoise_row_lengths(std::int64_t cond_tokens, std::int64_t uncond_tokens, const Guidance& guidance,
                                              std::int64_t plain_tokens) {
  // A vision-guided step without the cond prompt runs the plain one in its place, which is never longer.
  std::vector<std::int64_t> lengths = {std::max(cond_tokens, guidance.vision_plain ? plain_tokens : 0)};
  if (guidance.do_cfg()) lengths.push_back(uncond_tokens);
  if (guidance.slg_layers.any() && guidance.slg_scale > 0) lengths.push_back(cond_tokens);
  if (guidance.vision_two_rows()) lengths.push_back(plain_tokens);
  return lengths;
}

std::size_t denoise_loop_bytes(std::int64_t height, std::int64_t width, std::int64_t cond_tokens,
                               std::int64_t uncond_tokens, const Guidance& guidance, SamplerKind sampler,
                               std::int64_t plain_tokens) {
  const std::size_t rows = denoise_row_lengths(cond_tokens, uncond_tokens, guidance, plain_tokens).size();
  const std::int64_t numel = moss_dit::kLatentChannels * height * width;
  const std::size_t latent = runtime::align_up(static_cast<std::size_t>(numel) * sizeof(float));
  // One velocity per row, plus the driver's prediction and state.
  return rows * latent +
         SamplerDriver::workspace_bytes(sampler, numel, kMaxDenoiseSteps, static_cast<int>(moss_dit::kLatentChannels));
}

std::size_t denoise_workspace_bytes(std::int64_t height, std::int64_t width, std::int64_t cond_tokens,
                                    std::int64_t uncond_tokens, const Guidance& guidance, SamplerKind sampler,
                                    moss_dit::BlockPath path, std::int64_t plain_tokens) {
  return denoise_loop_bytes(height, width, cond_tokens, uncond_tokens, guidance, sampler, plain_tokens) +
         moss_dit::MossDit::workspace_bytes(height, width,
                                            denoise_row_lengths(cond_tokens, uncond_tokens, guidance, plain_tokens), path);
}

int denoise(moss_dit::MossDit& dit, runtime::DeviceArena& arena, const Tensor& cond_text, const Tensor& uncond_text,
            const Schedule& schedule, const Guidance& guidance, SamplerKind sampler, const SamplerOptions& options,
            const Tensor& latents, cudaStream_t stream, const StepCallback& on_step, const Tensor& plain_text) {
  using Clock = std::chrono::steady_clock;
  check_latent(latents, "latents");
  const std::int64_t h = latents.dim(1), w = latents.dim(2);
  const int num_steps = schedule.num_steps();
  if (num_steps < 1 || schedule.sigmas.size() != static_cast<std::size_t>(num_steps) + 1) {
    throw std::invalid_argument("pipeline::denoise: malformed schedule");
  }
  if (num_steps > kMaxDenoiseSteps) throw std::invalid_argument("pipeline::denoise: too many steps");
  const bool do_cfg = guidance.do_cfg();
  const bool slg_enabled = guidance.slg_layers.any() && guidance.slg_scale > 0;
  if (guidance.vision_plain && plain_text.data == nullptr) {
    throw std::invalid_argument("pipeline::denoise: vision guidance needs the plain prompt's refined text");
  }
  if (guidance.vision_plain && slg_enabled) throw std::invalid_argument("pipeline::denoise: vision guidance with an SLG row");

  SamplerOptions opts = options;
  opts.flow = schedule.flow;  // the schedule's shift drives the SDE samplers' first-sigma offset
  const bool official_euler = schedule.scheduler == SchedulerKind::kMoss;
  SamplerDriver driver(sampler, schedule.sigmas, {moss_dit::kLatentChannels, h, w}, opts, official_euler);

  runtime::StackScope scope(arena);
  const auto latent_tensor = [&] { return arena.alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, h, w}); };
  const Tensor v_cond = latent_tensor();
  const Tensor v_uncond = do_cfg ? latent_tensor() : Tensor{};
  const Tensor v_slg = slg_enabled ? latent_tensor() : Tensor{};
  const Tensor v_plain = guidance.vision_two_rows() ? latent_tensor() : Tensor{};

  const ModelFn model = [&](const Tensor& x, const ModelCall& call, const Tensor& out, cudaStream_t s) {
    const bool skip = guidance.skip_active(call.step, num_steps);
    // CFG truncation: past it the cond row runs alone (cfg_active is do_cfg when it is off).
    const bool cfg = guidance.cfg_active(call.step, num_steps);
    // Vision guidance: w 0 runs the plain prompt as the cond row, w 1 the cond prompt, else both, blended below.
    const double wv = guidance.vision_weight_at(call.step, num_steps);
    const bool both = wv != 0.0 && wv != 1.0;
    moss_dit::Row rows[4];
    Tensor outs[4];
    int n = 0;
    rows[n] = {wv == 0.0 ? plain_text : cond_text, {}};
    outs[n++] = v_cond;
    if (cfg) {
      rows[n] = {uncond_text, skip ? guidance.uncond_skip : moss_dit::SkipSet{}};
      outs[n++] = v_uncond;
    }
    const bool slg = slg_enabled && skip && cfg == do_cfg;
    if (slg) {
      rows[n] = {cond_text, guidance.slg_layers};
      outs[n++] = v_slg;
    }
    if (both) {
      rows[n] = {plain_text, {}};
      outs[n++] = v_plain;
    }
    dit.forward(x, call.timestep, std::span<const moss_dit::Row>(rows, n), std::span<const Tensor>(outs, n), s);
    if (both) cfg_combine(v_cond, v_plain, static_cast<float>(wv), v_cond, s);  // plain + w * (cond - plain)

    guide_prediction(driver.prediction(), x, call.sigma, v_cond, v_uncond, v_slg, cfg, static_cast<float>(guidance.cfg),
                     slg, static_cast<float>(guidance.slg_scale), out, s);
  };

  auto t0 = Clock::now();
  const StepFn after_step = [&](int finished) {
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto now = Clock::now();
    const StepInfo info{finished, num_steps, std::chrono::duration<double>(now - t0).count()};
    const bool go_on = !on_step || on_step(info);
    t0 = Clock::now();
    return go_on;
  };
  return driver.run(latents, model, arena, stream, after_step);
}

int denoise(moss_dit::MossDit& dit, runtime::DeviceArena& arena, const Tensor& cond_text, const Tensor& uncond_text,
            const Schedule& schedule, const Guidance& guidance, SamplerKind sampler, const Tensor& latents,
            cudaStream_t stream, const StepCallback& on_step) {
  if (sampler_needs_options(sampler)) {
    throw std::invalid_argument(std::string("pipeline::denoise: sampler ") + sampler_name(sampler) +
                                " needs SamplerOptions (seed or data-space parameters)");
  }
  return denoise(dit, arena, cond_text, uncond_text, schedule, guidance, sampler, SamplerOptions{}, latents, stream, on_step);
}

}  // namespace mi::pipeline
