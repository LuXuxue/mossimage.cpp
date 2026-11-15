// Vision guidance (pipeline.h Request::VisionReferences, sampler.h Guidance::vision_*): which steps the images guide
// and with what weight, the rows the loop sizes for, and the requests validate() refuses. The GPU effect (the images
// leave the composition to the prompt) is measured with the engine and recorded in docs/c-api.md#vision-references.
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "pipeline/pipeline.h"
#include "pipeline/sampler.h"

namespace {

using mi::pipeline::Guidance;
using mi::pipeline::Request;

Request with_image() {
  Request r;
  r.prompt = "a detective on a rooftop";
  auto im = std::make_shared<mi::pipeline::Image>();
  im->width = 32;
  im->height = 32;
  im->pixels.assign(32 * 32 * 3, 128);
  r.vision.images.push_back(im);
  return r;
}

TEST(VisionGuidance, OnlyAWeightOrWindowNeedsThePlainPrompt) {
  Request r = with_image();
  EXPECT_TRUE(r.vision.enabled());
  EXPECT_FALSE(r.vision.guided());  // 1, 0, 1: the images on every step, nothing extra
  r.vision.weight = 0.5f;
  EXPECT_TRUE(r.vision.guided());
  r.vision.weight = 1.0f;
  r.vision.start = 0.4;
  EXPECT_TRUE(r.vision.guided());
  r.vision.start = 0.0;
  r.vision.stop = 0.6;
  EXPECT_TRUE(r.vision.guided());
  // An empty window or weight 0 drops the images.
  r.vision.start = 0.6;
  EXPECT_FALSE(r.vision.enabled());
  EXPECT_FALSE(r.vision.guided());
  Request none = with_image();
  none.vision.weight = 0.0f;
  EXPECT_FALSE(none.vision.enabled());
}

TEST(VisionGuidance, WeightPerStepFollowsTheWindow) {
  Guidance g;
  EXPECT_EQ(g.vision_weight_at(0, 10), 1.0);  // no vision guidance: the cond prompt throughout
  g.vision_plain = true;
  g.vision_weight = 0.7;
  g.vision_start = 0.3;
  g.vision_stop = 0.8;
  std::vector<double> got;
  for (int i = 0; i < 10; ++i) got.push_back(g.vision_weight_at(i, 10));
  // start inclusive, stop exclusive, i / num_steps as slg_start / slg_stop.
  EXPECT_EQ(got, (std::vector<double>{0, 0, 0, 0.7, 0.7, 0.7, 0.7, 0.7, 0, 0}));
}

TEST(VisionGuidance, TwoCondRowsOnlyForAWeightBetween) {
  Guidance g;
  g.cfg = 3.0;
  g.vision_plain = true;
  g.vision_weight = 1.0;
  g.vision_start = 0.5;
  EXPECT_FALSE(g.vision_two_rows());
  // A window alone swaps the cond row's prompt: the largest row is the longer of the two.
  EXPECT_EQ(mi::pipeline::denoise_row_lengths(120, 20, g, 30), (std::vector<std::int64_t>{120, 20}));
  g.vision_weight = 0.5;
  EXPECT_TRUE(g.vision_two_rows());
  EXPECT_EQ(mi::pipeline::denoise_row_lengths(120, 20, g, 30), (std::vector<std::int64_t>{120, 20, 30}));
  g.cfg = 1.0;
  EXPECT_EQ(mi::pipeline::denoise_row_lengths(120, 0, g, 30), (std::vector<std::int64_t>{120, 30}));
  // Without vision guidance nothing changes.
  Guidance plain;
  EXPECT_EQ(mi::pipeline::denoise_row_lengths(120, 20, plain, 30), (std::vector<std::int64_t>{120, 20}));
}

TEST(VisionGuidance, ValidateRefusesABadWindowAndWhatCannotFollowTheRowSwap) {
  Request r = with_image();
  r.vision.start = 0.2;
  r.vision.stop = 0.9;
  EXPECT_NO_THROW(mi::pipeline::validate(r));
  Request backwards = r;
  backwards.vision.start = 0.95;
  EXPECT_THROW(mi::pipeline::validate(backwards), std::invalid_argument);
  Request past = r;
  past.vision.stop = 1.5;
  EXPECT_THROW(mi::pipeline::validate(past), std::invalid_argument);
  Request slg = r;
  slg.slg_scale = 2.0;
  EXPECT_THROW(mi::pipeline::validate(slg), std::invalid_argument);
  Request cached = r;
  cached.accel.step_cache = mi::pipeline::accel::StepCacheKind::kFbCache;
  EXPECT_THROW(mi::pipeline::validate(cached), std::invalid_argument);
  // Plain vision references (1, 0, 1) keep working with both.
  Request old = with_image();
  old.slg_scale = 2.0;
  old.slg_layers = {22};
  EXPECT_NO_THROW(mi::pipeline::validate(old));
}

}  // namespace
