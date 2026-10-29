#ifndef BIEVR_LIO_SYNCHRONIZER_H_
#define BIEVR_LIO_SYNCHRONIZER_H_

#include <deque>

#include "bievr_lio/pipeline.h"

namespace bievr {

class Synchronizer {
 public:
  explicit Synchronizer(std::shared_ptr<Pipeline> pipeline) : pipeline_(pipeline) {}
  virtual ~Synchronizer() = default;
  bool addImu(const ImuMeasurement& imu);
  bool addPointcloud(const StampedIntensityPointcloud& cloud);

 private:
  void synchronizeData();
  std::deque<ImuMeasurement> imu_queue_;
  std::deque<StampedIntensityPointcloud> point_queue_;
  uint64_t last_processed_end_stamp_ = 0;  // [Ultraview/Aegis] end_stamp of the last cloud sent to the pipeline
  std::shared_ptr<Pipeline> pipeline_;
};
}  // namespace bievr

#endif  // BIEVR_LIO_SYNCHRONIZER_H_