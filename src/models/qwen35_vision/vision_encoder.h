// Qwen3.5 vision tower (Qwen3_5VisionModel, transformers 5.x) as MOSS-Image's text encoder file carries it: images to
// tokens of the text encoder's width (4096) that take the place of `<|image_pad|>` placeholders in the prompt, so a
// reference image conditions the DiT through the prompt (vision references; tools/vision_ref has the reference).
//
// Architecture (the file's vision_config): Conv3d patch embedding 3x2x16x16 -> 1152, a learned 48x48 position table
// interpolated bilinearly to the patch grid, 27 blocks (LayerNorm, qkv with bias, 2-D rotary over 72-dim heads,
// bidirectional attention per image, GELU-tanh MLP 1152 -> 4304 -> 1152), then the 2x2 merger (LayerNorm, 4608 -> 4608
// GELU -> 4096). No DeepStack levels.
//
// Weights: Comfy-Org's INT8 ConvRot encoder stores the blocks' linears as INT8 ConvRot (group 64) and the rest in bf16.
// They are read once from the file into host memory (~0.5 GB). An encode uploads them as stored and dequantizes each
// linear on the GPU right before its GEMM, so the device holds ~0.5 GB of weights plus one bf16 linear at a time.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/tensor.h"
#include "pipeline/pipeline_image.h"
#include "runtime/device_arena.h"
#include "tokenizer/qwen_bpe.h"

namespace mi::qwen35_vision {

inline constexpr int kPatch = 16;
inline constexpr int kTemporal = 2;
inline constexpr int kMerge = 2;
inline constexpr int kDim = 1152;
inline constexpr int kMlp = 4304;
inline constexpr int kDepth = 27;
inline constexpr int kOut = 4096;
inline constexpr int kPosSide = 48;
inline constexpr int kPatchInput = 3 * kTemporal * kPatch * kPatch;  // 1536
inline constexpr int kGrid = kPatch * kMerge;                        // sides are multiples of 32

// Pixel cap per reference image when none is given: ~0.1 MP, about 100 prompt tokens. A character crop at this
// size carries the character without taking over the image (tools/vision_ref/README.md).
inline constexpr std::int64_t kDefaultMaxPixels = 100'000;

// The size an image is resized to before the tower: within max_pixels keeping the aspect ratio, never upscaled,
// each side rounded to a multiple of 32 (at least 32). {width, height}.
std::pair<int, int> fit_size(int width, int height, std::int64_t max_pixels);

// One image as the tower sees it: patch rows and their grid, in the merger's block order.
struct Prepared {
  int grid_h = 0, grid_w = 0;                // patch grid (16-px patches)
  std::vector<std::uint16_t> patches;        // bf16 bits [grid_h * grid_w, 1536]: (C, T, P, P), normalized
  std::int64_t tokens() const { return static_cast<std::int64_t>(grid_h) * grid_w / (kMerge * kMerge); }
};

// What a set of prepared images contributes to an embedding cache key: every image's patch grid and patch values,
// in order. The grid matters too: two images can have the same patch values on grids of other shapes (a flat
// colour at 64x128 and at 128x64), and their position embeddings and mRoPE positions differ.
std::uint64_t images_hash(const std::vector<Prepared>& images, std::uint64_t seed);

// RGB image -> Prepared: Lanczos resize to fit_size, rescale to [0, 1], `weight` (0.5 + weight * (x - 0.5): < 1 fades
// the image toward gray, which the tower reads as no signal), normalize with mean = std = 0.5, the temporal pair (a
// still image is its own second frame), and the patch order of Qwen2VLImageProcessor.
Prepared prepare(const pipeline::Image& rgb, std::int64_t max_pixels, float weight);

// Rotary cos / sin of the patches of one grid, f32 [N, 72] each (block-major positions, Qwen3_5VisionRotaryEmbedding).
void rotary_tables(int grid_h, int grid_w, std::vector<float>& cos, std::vector<float>& sin);

// The interleaved mRoPE positions of a prompt with images (Qwen3_5Model.get_rope_index for images): text tokens
// count up on all three axes; an image's tokens take (t, h, w) = (s, s + row, s + col) on its merged grid, s being the
// next text position, and the text after it continues from the image's largest position + 1.
// `image_starts[i]` is the index of image i's first placeholder in the sequence; grids are the merged token grids.
// Returns [3][length] positions (t, h, w).
std::vector<std::vector<std::int32_t>> mrope_positions(std::int64_t length, const std::vector<std::int64_t>& image_starts,
                                                       const std::vector<std::pair<int, int>>& merged_grids);

// A MOSS prompt with images in it, as ComfyUI-MOSSImageTextEncoder builds it: each image's
// `<|vision_start|><|image_pad|>...<|vision_end|>` (one pad per merged token) right after "<|im_start|>user\n", in
// order, then the prompt and the template's end.
struct VisionPrompt {
  std::vector<std::int32_t> ids;
  std::vector<std::int64_t> starts;                 // index of each image's first pad
  std::vector<std::pair<int, int>> merged_grids;    // each image's token grid (patch grid / 2)
  std::vector<std::int32_t> positions;              // mRoPE [3 * ids.size()]: t, then h, then w
};
// Throws std::invalid_argument when the prompt with its images is longer than max_length (truncating would cut an
// image apart) or the prompt is one the template rejects.
VisionPrompt build_prompt(const tok::Tokenizer& tokenizer, std::string_view prompt, const std::vector<Prepared>& images,
                          std::size_t max_length);

class VisionEncoder {
 public:
  // True when `te_file` (a Qwen3.5 text encoder safetensors file) carries the vision tower.
  static bool available(const std::filesystem::path& te_file);

  // Reads the tower's tensors (and only those) into host memory and checks their shapes. Throws std::runtime_error
  // naming the tensor for a file without the tower or with an unexpected layout.
  explicit VisionEncoder(const std::filesystem::path& te_file);
  ~VisionEncoder();
  VisionEncoder(const VisionEncoder&) = delete;
  VisionEncoder& operator=(const VisionEncoder&) = delete;

  // Device bytes encode() takes from the arena stack for `patches` patches in all (weights, scratch and the output),
  // an upper bound from the architecture (the plan's figure before the file is read).
  static std::size_t workspace_bytes(std::int64_t patches);

  // The tokens of every image, bf16 [sum of tokens(), 4096], in order. The output is allocated on the arena stack
  // below encode()'s scratch, which is released before returning (TextEncoder::encode's convention). Synchronizes
  // `stream` before returning.
  Tensor encode(const std::vector<Prepared>& images, runtime::DeviceArena& arena, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mi::qwen35_vision
