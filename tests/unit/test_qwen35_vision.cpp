// Qwen3.5 vision tower (models/qwen35_vision): the host-side geometry against transformers 5.x, and, with the encoder
// file and a reference dump of tools/vision_ref/vision_conditioning.py (--dump-vision), the GPU tower against
// transformers' Qwen3_5VisionModel on the same image:
//   MOSS_VISION_TE=<qwen3.5_9b_int8_convrot.safetensors> MOSS_VISION_IMAGE=<png> MOSS_VISION_REF=<dump.safetensors>
// (the dump made with --megapixels 100000/1048576, the encoder's default cap).
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "models/qwen35_vision/vision_encoder.h"
#include "pipeline/pipeline_image.h"
#include "runtime/device_arena.h"
#include "weights/safetensors.h"

namespace {

using namespace mi::qwen35_vision;

float bf16_to_float(std::uint16_t v) {
  std::uint32_t u = static_cast<std::uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

TEST(Qwen35Vision, FitSizeKeepsTheAspectRatioOnTheMergeGridAndNeverUpscales) {
  EXPECT_EQ(fit_size(460, 500, 100'000), std::make_pair(288, 320));
  EXPECT_EQ(fit_size(1200, 864, 1'048'576), std::make_pair(1184, 864));  // snapping to 1216 would pass the cap
  EXPECT_EQ(fit_size(64, 64, 100'000), std::make_pair(64, 64));
  EXPECT_EQ(fit_size(10, 10, 100'000), std::make_pair(32, 32));  // at least one merged token
  const auto [w, h] = fit_size(4000, 1000, 100'000);
  EXPECT_LE(w * h, 100'000);
  EXPECT_EQ(w % 32, 0);
  EXPECT_EQ(h % 32, 0);
  EXPECT_THROW(fit_size(0, 10, 100), std::invalid_argument);
}

TEST(Qwen35Vision, ImagesHashTellsGridsApartWithTheSamePatches) {
  // A flat colour at 64x128 and at 128x64: the same 32 patch rows, other position embeddings and mRoPE positions.
  Prepared tall, wide;
  tall.grid_h = 8;
  tall.grid_w = 4;
  wide.grid_h = 4;
  wide.grid_w = 8;
  tall.patches.assign(32u * kPatchInput, 0x3f80);
  wide.patches = tall.patches;
  EXPECT_NE(images_hash({tall}, 7), images_hash({wide}, 7));
  EXPECT_EQ(images_hash({tall}, 7), images_hash({tall}, 7));
  EXPECT_NE(images_hash({tall, wide}, 7), images_hash({wide, tall}, 7));  // order counts
  EXPECT_NE(images_hash({tall}, 7), images_hash({tall}, 8));              // the model's own identity is kept
}

TEST(Qwen35Vision, AvailableOnlyForAFileWithTheTower) {
  // A minimal safetensors file holding one text-encoder tensor and no model.visual.*: not usable for vision references.
  const auto dir = std::filesystem::temp_directory_path() / ("mi_vision_avail_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  std::filesystem::create_directories(dir);
  const auto write = [&](const std::string& name, const std::string& tensor) {
    const std::string header = "{\"" + tensor + "\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}";
    std::ofstream o(dir / name, std::ios::binary);
    const std::uint64_t n = header.size();
    o.write(reinterpret_cast<const char*>(&n), 8);
    o << header;
    o.write("\0\0\0\0", 4);
    return dir / name;
  };
  EXPECT_FALSE(VisionEncoder::available(write("text_only.safetensors", "model.language_model.norm.weight")));
  EXPECT_TRUE(VisionEncoder::available(write("with_tower.safetensors", "model.visual.merger.norm.bias")));
  EXPECT_FALSE(VisionEncoder::available(dir / "absent.safetensors"));
  std::filesystem::remove_all(dir);
}

TEST(Qwen35Vision, MropePositionsMatchTransformersGetRopeIndex) {
  // Qwen3_5Model.get_rope_index for ids [t t t img*6 t t img*2 t], image grids (1, 4, 6) and (1, 2, 4).
  const auto pos = mrope_positions(14, {3, 11}, {{2, 3}, {1, 2}});
  const std::vector<std::vector<std::int32_t>> expected = {
      {0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8, 8, 10},
      {0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8, 8, 10},
      {0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10},
  };
  EXPECT_EQ(pos, expected);
  const auto text = mrope_positions(4, {}, {});
  EXPECT_EQ(text[0], (std::vector<std::int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(text[0], text[2]);
  EXPECT_THROW(mrope_positions(5, {3}, {{2, 2}}), std::invalid_argument);  // the image runs past the end
}

TEST(Qwen35Vision, RotaryTablesAreBlockMajorHeightThenWidth) {
  std::vector<float> cos, sin;
  rotary_tables(4, 4, cos, sin);
  ASSERT_EQ(cos.size(), 16u * 72u);
  // Patch 0 is (0, 0): no rotation.
  for (int i = 0; i < 72; ++i) {
    EXPECT_FLOAT_EQ(cos[static_cast<std::size_t>(i)], 1.0f);
    EXPECT_FLOAT_EQ(sin[static_cast<std::size_t>(i)], 0.0f);
  }
  // Patch 1 in block order is (0, 1): the height half is unrotated, the width half rotated by inv_freq[0] = 1.
  EXPECT_FLOAT_EQ(cos[72 + 0], 1.0f);
  EXPECT_FLOAT_EQ(cos[72 + 18], std::cos(1.0f));
  EXPECT_FLOAT_EQ(sin[72 + 36 + 18], std::sin(1.0f));  // the second copy of the 36
  // Patch 2 is (1, 0), the first patch of the block's second row.
  EXPECT_FLOAT_EQ(cos[2 * 72 + 0], std::cos(1.0f));
  EXPECT_FLOAT_EQ(cos[2 * 72 + 18], 1.0f);
}

TEST(Qwen35Vision, TowerMatchesTransformersOnARealImage) {
  const char* te = std::getenv("MOSS_VISION_TE");
  const char* image = std::getenv("MOSS_VISION_IMAGE");
  const char* ref = std::getenv("MOSS_VISION_REF");
  if (!te || !image || !ref) GTEST_SKIP() << "set MOSS_VISION_TE, MOSS_VISION_IMAGE and MOSS_VISION_REF";

  const Prepared prepared = prepare(mi::pipeline::load_image(image), kDefaultMaxPixels, 1.0f);
  mi::weights::SafetensorsFile f(ref);
  std::ifstream in(ref, std::ios::binary);
  auto read = [&](const char* name) {
    const auto* t = f.find(name);
    EXPECT_NE(t, nullptr) << name;
    std::vector<std::uint16_t> v(t->nbytes / 2);
    in.seekg(static_cast<std::streamoff>(t->offset));
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(t->nbytes));
    return v;
  };

  // The processor's patches: same resize (PIL Lanczos), normalization and order, so the same bf16 bits.
  const auto ref_patches = read("patches");
  ASSERT_EQ(ref_patches.size(), prepared.patches.size());
  EXPECT_EQ(ref_patches, prepared.patches);

  VisionEncoder encoder(te);
  mi::runtime::DeviceArena arena(encoder.workspace_bytes(static_cast<std::int64_t>(prepared.grid_h) * prepared.grid_w) +
                                 (std::size_t{64} << 20));
  cudaStream_t stream = nullptr;
  const mi::Tensor out = encoder.encode({prepared}, arena, stream);
  ASSERT_EQ(out.dim(0), prepared.tokens());
  std::vector<std::uint16_t> got(static_cast<std::size_t>(out.numel()));
  ASSERT_EQ(cudaMemcpy(got.data(), out.data, got.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
  const auto want = read("vision");
  ASSERT_EQ(want.size(), got.size());

  // Not bit-exact (cuBLASLt vs torch GEMMs, our fused kernels). bf16 through 27 blocks drifts by itself: transformers'
  // own bf16 tower is within cosine 0.990 (worst token) and 3.2 % relative L2 of its fp32 run on this kind of image.
  // So the engine is held to that against the fp32 truth when given (MOSS_VISION_TRUTH, raw f32 [N, 4096] from the
  // fp32 tower), and to twice it against transformers' bf16 tokens.
  auto compare = [&](auto&& want_at, double& worst, double& rel) {
    double num = 0, den = 0;
    worst = 1.0;
    for (std::int64_t t = 0; t < out.dim(0); ++t) {
      double dot = 0, a2 = 0, b2 = 0;
      for (int d = 0; d < kOut; ++d) {
        const double a = bf16_to_float(got[static_cast<std::size_t>(t * kOut + d)]);
        const double b = want_at(static_cast<std::size_t>(t * kOut + d));
        dot += a * b;
        a2 += a * a;
        b2 += b * b;
        num += (a - b) * (a - b);
        den += b * b;
      }
      worst = std::min(worst, dot / std::sqrt(a2 * b2));
    }
    rel = std::sqrt(num / den);
  };
  double worst = 0, rel = 0;
  compare([&](std::size_t i) { return static_cast<double>(bf16_to_float(want[i])); }, worst, rel);
  EXPECT_GT(worst, 0.975) << "worst token cosine against transformers bf16";
  EXPECT_LT(rel, 0.065) << "relative L2 against transformers bf16";
  RecordProperty("bf16_worst_cosine", std::to_string(worst));
  RecordProperty("bf16_relative_l2", std::to_string(rel));
  if (const char* truth_path = std::getenv("MOSS_VISION_TRUTH")) {
    std::vector<float> truth(got.size());
    std::ifstream tf(truth_path, std::ios::binary);
    tf.read(reinterpret_cast<char*>(truth.data()), static_cast<std::streamsize>(truth.size() * 4));
    ASSERT_TRUE(tf.good()) << "MOSS_VISION_TRUTH must hold f32 [tokens, 4096]";
    compare([&](std::size_t i) { return static_cast<double>(truth[i]); }, worst, rel);
    EXPECT_GT(worst, 0.985) << "worst token cosine against the fp32 tower";
    EXPECT_LT(rel, 0.04) << "relative L2 against the fp32 tower";
    std::printf("vision tower vs fp32 truth: worst cosine %.4f, relative L2 %.4f\n", worst, rel);
  }
}

}  // namespace

// The weights are the same for every grid, so the growth of the arena's peak between a small and a large grid is what
// encode() really needs per patch; the estimate must grow at least as much. (An INT8 file needs less for its weights
// than the estimate counts, which would hide a short activation count in a single run.)
TEST(Qwen35Vision, WorkspaceEstimateCoversWhatEncodeHoldsPerPatch) {
  const char* te = std::getenv("MOSS_VISION_TE");
  if (!te) GTEST_SKIP() << "set MOSS_VISION_TE";
  auto gray = [](int side) {
    mi::pipeline::Image image;
    image.width = image.height = side;
    image.pixels.assign(static_cast<std::size_t>(side) * side * 3, 128);
    return image;
  };
  const Prepared small = prepare(gray(256), kDefaultMaxPixels, 1.0f);
  const Prepared large = prepare(gray(2048), 2048 * 2048, 1.0f);
  const auto patches = [](const Prepared& p) { return static_cast<std::int64_t>(p.grid_h) * p.grid_w; };
  ASSERT_GE(patches(large), 16 * patches(small));

  VisionEncoder encoder(te);
  mi::runtime::DeviceArena arena(VisionEncoder::workspace_bytes(patches(large)) + (std::size_t{64} << 20));
  cudaStream_t stream = nullptr;
  auto peak = [&](const Prepared& p) {
    const std::size_t mark = arena.mark();
    arena.reset_high_water();
    const std::size_t before = arena.used();
    (void)encoder.encode({p}, arena, stream);
    const std::size_t used = arena.high_water() - before;
    arena.rollback(mark);
    return used;
  };
  const std::size_t small_peak = peak(small), large_peak = peak(large);
  EXPECT_LE(large_peak - small_peak,
            VisionEncoder::workspace_bytes(patches(large)) - VisionEncoder::workspace_bytes(patches(small)));
}

#include "models/qwen35_text/quant_text_model.h"
#include "models/qwen35_text/text_encoder.h"

namespace {

// The whole conditioning: tokens, the tower, the image rows and mRoPE in the text encoder, against transformers'
// Qwen3_5Model (tools/vision_ref/vision_conditioning.py --prompt x --image <MOSS_VISION_IMAGE> --megapixels 0.095,
// written to MOSS_VISION_COND), with the bundle's tokenizer.json in MOSS_VISION_TOKENIZER.
TEST(Qwen35Vision, ConditioningMatchesTransformersQwen3_5Model) {
  const char* te = std::getenv("MOSS_VISION_TE");
  const char* image = std::getenv("MOSS_VISION_IMAGE");
  const char* cond = std::getenv("MOSS_VISION_COND");
  const char* tokenizer_json = std::getenv("MOSS_VISION_TOKENIZER");
  if (!te || !image || !cond || !tokenizer_json) GTEST_SKIP() << "set MOSS_VISION_TE, _IMAGE, _COND and _TOKENIZER";
  const auto tokenizer = mi::tok::Tokenizer::load(tokenizer_json);
  const Prepared prepared = prepare(mi::pipeline::load_image(image), kDefaultMaxPixels, 1.0f);
  const char* prompt_env = std::getenv("MOSS_VISION_PROMPT");
  const std::string prompt = prompt_env ? prompt_env : "x";
  const VisionPrompt vp = build_prompt(tokenizer, prompt, {prepared}, 4096);
  ASSERT_EQ(vp.starts.size(), 1u);
  EXPECT_EQ(vp.ids[static_cast<std::size_t>(vp.starts[0]) - 1], tokenizer.encode("<|vision_start|>")[0]);

  mi::qwen35_text::QuantTextModel model(te);
  VisionEncoder vision(te);
  const std::vector<std::int64_t> lengths{static_cast<std::int64_t>(vp.ids.size())};
  mi::qwen35_text::EncoderOptions eo;
  mi::runtime::DeviceArena arena(mi::qwen35_text::TextEncoder::workspace_bytes(model, lengths, eo) +
                                 mi::qwen35_text::TextEncoder::vision_workspace_bytes(lengths) +
                                 vision.workspace_bytes(static_cast<std::int64_t>(prepared.grid_h) * prepared.grid_w));
  cudaStream_t stream = nullptr;
  mi::runtime::StackScope scope(arena);
  const mi::Tensor rows = vision.encode({prepared}, arena, stream);
  mi::qwen35_text::TextEncoder encoder(model, arena, stream, eo);
  std::vector<mi::qwen35_text::SequenceVision> sv(1);
  sv[0].rows.emplace_back(vp.starts[0], rows);
  sv[0].positions = vp.positions;
  mi::qwen35_text::EncodeOptions opts;
  opts.vision = &sv;
  const auto got = encoder.encode_to_host({vp.ids}, opts)[0];
  if (const char* dump = std::getenv("MOSS_VISION_DUMP")) {  // the engine's conditioning, for drawing it
    const std::string header = "{\"conditioning\":{\"dtype\":\"BF16\",\"shape\":[" + std::to_string(got.size() / 4096) +
                               ",4096],\"data_offsets\":[0," + std::to_string(got.size() * 2) + "]}}";
    std::ofstream o(dump, std::ios::binary);
    const std::uint64_t n = header.size();
    o.write(reinterpret_cast<const char*>(&n), 8);
    o << header;
    o.write(reinterpret_cast<const char*>(got.data()), static_cast<std::streamsize>(got.size() * 2));
  }
  mi::weights::SafetensorsFile f(cond);
  const auto* t = f.find("conditioning");
  ASSERT_NE(t, nullptr);
  ASSERT_EQ(t->nbytes, got.size() * 2) << "token counts differ";
  std::vector<std::uint16_t> want(got.size());
  std::ifstream in(cond, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(t->offset));
  in.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(t->nbytes));
  // The engine runs the decoder's GEMMs in INT8 with per-token activations, the reference in bf16 with dequantized
  // weights. Text tokens agree to cosine 0.998 as on text-only prompts; a few image tokens, whose activations have
  // larger outliers, drift further (worst 0.69 on the test image) without changing the image drawn from them: the
  // engine's and transformers' conditionings give the same composition and character (checked by eye, PSNR 15 dB).
  // So the bound is on the mean and the overall error, not the worst token.
  double worst = 1, mean = 0, num = 0, den = 0;
  const std::size_t n = got.size() / mi::qwen35_text::kHidden;
  for (std::size_t r = 0; r < n; ++r) {
    double dot = 0, a2 = 0, b2 = 0;
    for (int d = 0; d < mi::qwen35_text::kHidden; ++d) {
      const double a = bf16_to_float(got[r * mi::qwen35_text::kHidden + static_cast<std::size_t>(d)]);
      const double b = bf16_to_float(want[r * mi::qwen35_text::kHidden + static_cast<std::size_t>(d)]);
      dot += a * b;
      a2 += a * a;
      b2 += b * b;
      num += (a - b) * (a - b);
      den += b * b;
    }
    worst = std::min(worst, dot / std::sqrt(a2 * b2));
    mean += dot / std::sqrt(a2 * b2) / static_cast<double>(got.size() / mi::qwen35_text::kHidden);
  }
  std::printf("conditioning vs transformers: %zu tokens, mean cosine %.4f, worst %.4f, relative L2 %.4f\n", n, mean, worst,
              std::sqrt(num / den));
  EXPECT_GT(mean, 0.97);
  EXPECT_LT(std::sqrt(num / den), 0.25);
}

}  // namespace

namespace {

// Text-only sequences through the vision path (mRoPE table from equal positions, no rows) give the plain path's bits:
// prompts without images are untouched by the feature.
TEST(Qwen35Vision, TextOnlyThroughTheVisionPathIsBitIdentical) {
  const char* te = std::getenv("MOSS_VISION_TE");
  const char* tokenizer_json = std::getenv("MOSS_VISION_TOKENIZER");
  if (!te || !tokenizer_json) GTEST_SKIP() << "set MOSS_VISION_TE and MOSS_VISION_TOKENIZER";
  const auto tokenizer = mi::tok::Tokenizer::load(tokenizer_json);
  const VisionPrompt vp = build_prompt(tokenizer, "一只橘猫坐在雨夜的木箱上", {}, 4096);
  EXPECT_EQ(vp.ids, mi::tok::encode_moss_prompt(tokenizer, "一只橘猫坐在雨夜的木箱上", 4096));
  mi::qwen35_text::QuantTextModel model(te);
  const std::vector<std::int64_t> lengths{static_cast<std::int64_t>(vp.ids.size())};
  mi::runtime::DeviceArena arena(mi::qwen35_text::TextEncoder::workspace_bytes(model, lengths, {}) +
                                 mi::qwen35_text::TextEncoder::vision_workspace_bytes(lengths));
  mi::qwen35_text::TextEncoder encoder(model, arena, nullptr, {});
  const auto plain = encoder.encode_to_host({vp.ids}, {})[0];
  std::vector<mi::qwen35_text::SequenceVision> sv(1);
  sv[0].positions = vp.positions;
  mi::qwen35_text::EncodeOptions opts;
  opts.vision = &sv;
  EXPECT_EQ(encoder.encode_to_host({vp.ids}, opts)[0], plain);
}

}  // namespace
