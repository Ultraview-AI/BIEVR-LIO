#ifndef BIEVR_LIO_LS_OPTIMIZER_H_
#define BIEVR_LIO_LS_OPTIMIZER_H_

#include <memory>

#include "bievr_lio/bievr_map.h"
#include "bievr_lio/common.h"

namespace bievr {

using Matrix66 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Row6 = Eigen::Matrix<double, 1, 6, Eigen::RowMajor>;

struct RegistrationConfig {
  double lm_init_lambda_factor = 1e-9;
  double rotation_epsilon = 1e-4;
  double transformation_epsilon = 1e-5;
  double huber_delta = 0.1;
  int max_iterations = 20;
  int lm_max_iterations = 20;
  bool lm_debug_print = false;
  bool img_residual = true;
  bool img_jacobian = true;
  // [Ultraview/Aegis] the cost-only evaluations of an LM iteration reuse the voxel each point was associated
  // with in that iteration's Jacobian linearization (as nanoGICP/fast_gicp: correspondences are updated in
  // linearize, not in the error evaluation): no per-point hash / 26-neighbour search per trial step, and the
  // accept/reject ratio compares the same residual set
  bool fixed_correspondences = false;
  // [Ultraview/Aegis] degeneracy-aware update (solution remapping, Zhang et al. ICRA 2016), on the translation only.
  // The geometric information on translation is the mean over the effective points of n n^T (n = the voxel's plane
  // normal, in the frame of the update; trace 1). Along its eigenvectors below this value the surfaces in view do not
  // pin the translation, and only the bump-image gradient would move it: the update is projected out there, so the
  // IMU prediction is kept. 0 = off (stock). Aegis 2026-10-31 (MAP=737out replay): floor-only takeoff scans with the
  // -45 deg OS0 have ~0.005-0.01 along body x and the image term turned the climb into forward motion (3.3 m); a
  // door wall 3 m ahead gives 0.13.
  double degeneracy_min_info = 0.0;
};

inline bool getSubPixelValue(const Voxel* voxel, const double x, const double y, double& value) {
  int x0 = std::floor(x);
  int y0 = std::floor(y);

  int x1 = x0 + 1;
  int y1 = y0 + 1;

  const int max_x = voxel->bump_smoothed_.cols() - 1;
  const int max_y = voxel->bump_smoothed_.rows() - 1;
  if (x0 < 0 || y0 < 0 || x1 > max_x || y1 > max_y) return false;

  const auto& M = voxel->bump_smoothed_;
  const auto& V = voxel->bump_weights_;

  double dx = x - x0;
  double dy = y - y0;
  double dx1 = 1.0 - dx;
  double dy1 = 1.0 - dy;

  int v00 = V(y0, x0) > 0, v01 = V(y0, x1) > 0, v10 = V(y1, x0) > 0, v11 = V(y1, x1) > 0;

  double w0 = dx1 * dy1 * v00;
  double w1 = dx * dy1 * v01;
  double w2 = dx1 * dy * v10;
  double w3 = dx * dy * v11;

  double wsum = w0 + w1 + w2 + w3;
  if (wsum == 0.0f) return false;

  value = (w0 * M(y0, x0) + w1 * M(y0, x1) + w2 * M(y1, x0) + w3 * M(y1, x1)) / wsum;
  return true;
}

// Combined bilinear sample + central-difference gradient.
// Shares floor/bounds/fraction work and reuses the 4 inner corner reads.
// Returns false if the center 2x2 stencil is out of bounds or has no valid corners.
// Gradients are set to 0 when their 4x2 stencil is out of bounds or has no valid corners.
inline bool sampleValueAndGradient(const Voxel* voxel, const double x, const double y,
                                   double& value, double& dIdx, double& dIdy) {
  const int x0 = std::floor(x);
  const int y0 = std::floor(y);
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;

  const int max_x = voxel->bump_smoothed_.cols() - 1;
  const int max_y = voxel->bump_smoothed_.rows() - 1;
  if (x0 < 0 || y0 < 0 || x1 > max_x || y1 > max_y) return false;

  const auto& M = voxel->bump_smoothed_;
  const auto& V = voxel->bump_weights_;

  const double dx = x - x0;
  const double dy = y - y0;
  const double dx1 = 1.0 - dx;
  const double dy1 = 1.0 - dy;

  const int v00 = V(y0, x0) > 0, v01 = V(y0, x1) > 0;
  const int v10 = V(y1, x0) > 0, v11 = V(y1, x1) > 0;
  {
    const double w0 = dx1 * dy1 * v00;
    const double w1 = dx * dy1 * v01;
    const double w2 = dx1 * dy * v10;
    const double w3 = dx * dy * v11;
    const double ws = w0 + w1 + w2 + w3;
    if (ws == 0.0) return false;
    value = (w0 * M(y0, x0) + w1 * M(y0, x1) + w2 * M(y1, x0) + w3 * M(y1, x1)) / ws;
  }

  dIdx = 0.0;
  dIdy = 0.0;

  if (x0 >= 1 && x1 + 1 <= max_x) {
    const int xm = x0 - 1;
    const int xp = x1 + 1;

    const int am0 = V(y0, xm) > 0, am2 = V(y1, xm) > 0;
    const double wm0 = dx1 * dy1 * am0;
    const double wm1 = dx * dy1 * v00;
    const double wm2 = dx1 * dy * am2;
    const double wm3 = dx * dy * v10;
    const double wms = wm0 + wm1 + wm2 + wm3;

    const int ap1 = V(y0, xp) > 0, ap3 = V(y1, xp) > 0;
    const double wp0 = dx1 * dy1 * v01;
    const double wp1 = dx * dy1 * ap1;
    const double wp2 = dx1 * dy * v11;
    const double wp3 = dx * dy * ap3;
    const double wps = wp0 + wp1 + wp2 + wp3;

    if (wms > 0.0 && wps > 0.0) {
      const double vm =
          (wm0 * M(y0, xm) + wm1 * M(y0, x0) + wm2 * M(y1, xm) + wm3 * M(y1, x0)) / wms;
      const double vp =
          (wp0 * M(y0, x1) + wp1 * M(y0, xp) + wp2 * M(y1, x1) + wp3 * M(y1, xp)) / wps;
      dIdx = 0.5 * (vp - vm);
    }
  }

  if (y0 >= 1 && y1 + 1 <= max_y) {
    const int ym = y0 - 1;
    const int yp = y1 + 1;

    const int am0 = V(ym, x0) > 0, am1 = V(ym, x1) > 0;
    const double wm0 = dx1 * dy1 * am0;
    const double wm1 = dx * dy1 * am1;
    const double wm2 = dx1 * dy * v00;
    const double wm3 = dx * dy * v01;
    const double wms = wm0 + wm1 + wm2 + wm3;

    const int ap2 = V(yp, x0) > 0, ap3 = V(yp, x1) > 0;
    const double wp0 = dx1 * dy1 * v10;
    const double wp1 = dx * dy1 * v11;
    const double wp2 = dx1 * dy * ap2;
    const double wp3 = dx * dy * ap3;
    const double wps = wp0 + wp1 + wp2 + wp3;

    if (wms > 0.0 && wps > 0.0) {
      const double vm =
          (wm0 * M(ym, x0) + wm1 * M(ym, x1) + wm2 * M(y0, x0) + wm3 * M(y0, x1)) / wms;
      const double vp =
          (wp0 * M(y1, x0) + wp1 * M(y1, x1) + wp2 * M(yp, x0) + wp3 * M(yp, x1)) / wps;
      dIdy = 0.5 * (vp - vm);
    }
  }

  return true;
}

struct Accumulator {
  int count = 0;
  double error_sum = 0.0;
  Matrix66 H = Matrix66::Zero();
  Vector6 b = Vector6::Zero();
  M3 H_geo_t = M3::Zero();  // [Ultraview/Aegis] geometric information on translation (normals only)
  double w_sum = 0.0;
  double huber_delta = 0.2;  // default delta

  // n_t: the plane normal as seen by the translation update (the geometric part of J's translation block)
  inline void add(double r, const Row6* J, const V3* n_t = nullptr) {
    ++count;
    double abs_r = std::abs(r);
    bool inlier = abs_r <= huber_delta;
    double w = inlier ? 1.0 : huber_delta / abs_r;

    error_sum += inlier ? 0.5 * r * r : huber_delta * (abs_r - 0.5 * huber_delta);

    if (J) {
      const Vector6 wJ = w * J->transpose();
      H.noalias() += wJ * (*J);
      b.noalias() += wJ * r;
    }
    if (n_t) {
      H_geo_t.noalias() += w * (*n_t) * n_t->transpose();
      w_sum += w;
    }
  }

  inline void merge(const Accumulator& other) {
    count += other.count;
    error_sum += other.error_sum;
    H += other.H;
    b += other.b;
    H_geo_t += other.H_geo_t;
    w_sum += other.w_sum;
  }
};

class LsqRegistration {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LsqRegistration(const BIEVRMap& map, const Pointcloud& source,
                  const RegistrationConfig& config = RegistrationConfig());
  virtual ~LsqRegistration() = default;

  Transform computeTransformation(const Transform& T_W_L_init);

  // Number of source points that found a valid map correspondence in the last
  // linearization with Jacobians (i.e. the points that actually constrained the
  // pose). Reported on the dashboard as "Effective Points".
  int numEffectivePoints() const { return num_effective_points_; }

  // [Ultraview/Aegis] degeneracy (last Jacobian linearization): the smallest eigenvalue of the geometric information on
  // translation (0..1/3), its eigenvector (frame of the update: the IMU/body frame), and how many translation
  // directions were projected out (0 when degeneracy_min_info is 0)
  double minTranslationInfo() const { return min_info_; }
  const V3& weakestDirection() const { return weakest_dir_; }
  int numDegenerateDirections() const { return n_degenerate_; }

 private:
  bool isConverged(const Transform& delta) const;

  double linearize(const Transform& T_W_L, Matrix66* H = nullptr, Vector6* b = nullptr);
  void updateDegeneracy(const Accumulator& total);

  bool stepLm(Transform& x0, Transform& delta);

  RegistrationConfig config_;
  double lm_lambda_ = -1.0;
  const BIEVRMap& map_;
  const Pointcloud& points_j_;
  std::vector<M3> skew_points_j_;
  bool converged_ = false;
  int num_effective_points_ = 0;
  std::vector<const Voxel*> voxel_of_;  // per source point, from the last Jacobian linearization
  M3 P_t_ = M3::Identity();  // projector onto the well-constrained translation directions
  double min_info_ = 0.0;
  V3 weakest_dir_ = V3::Zero();
  int n_degenerate_ = 0;
};

}  // namespace bievr
#endif  // BIEVR_LIO_LS_OPTIMIZER_H_