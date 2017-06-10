#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */


  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = n_x_ + 2;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  n_sigma_ = 2 * n_aug_ + 1;

  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

  ///* Weights of sigma points
  weights_ = VectorXd(n_sigma_);
  weights_.fill(1. / (2. * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  Xsig = MatrixXd(n_aug_, n_sigma_);

  n_z_radar_ = 3;

  n_z_laser_ = 2;

  R_radar_ = MatrixXd::Zero(n_z_radar_, n_z_radar_);

  R_laser_ = MatrixXd::Zero(n_z_laser_, n_z_laser_);

  H_ = MatrixXd(n_z_laser_, n_x_);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if ((!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) ||
      (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR))
    return;

  if (!is_initialized_) {
    // initial timestamp
    time_us_ = meas_package.timestamp_;

    x_ << 0, 0, 0, 0, 0;

    H_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

    // set R noise matrices
    R_laser_(0, 0) = pow(std_laspx_, 2);
    R_laser_(1, 1) = pow(std_laspy_, 2);
    R_radar_(0, 0) = pow(std_radr_, 2);
    R_radar_(1, 1) = pow(std_radphi_, 2);
    R_radar_(2, 2) = pow(std_radrd_, 2);

    // initialize state covariance matrix
    P_ = MatrixXd::Identity(n_x_, n_x_);

    double px = 0., py = 0.;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_d = meas_package.raw_measurements_[2];

      px = rho * cos(phi);
      py = rho * sin(phi);
    }

    x_[0] = px;
    x_[1] = py;

    is_initialized_ = true;

    return;
  }

  // prediction step
  double delta_t = (meas_package.timestamp_ - time_us_) / 1.e6;

  Prediction(delta_t);
  time_us_ = meas_package.timestamp_;

  // update step, with the corresponding sensor
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  CreateAugmentedSigmaPoints();

  PredictAugmentedSigmaPoints(delta_t);

  // predict state mean
  x_.fill(0.0);
  // iterate over sigma points
  for (int i = 0; i < n_sigma_; i++) {
    x_ += weights_[i] * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff[3] = NormalizeAngle(x_diff[3]);
    P_ += weights_[i] * x_diff * x_diff.transpose();
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_laser_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  x_ = x_ + K * z_diff;
  P_ = (I - K * H_) * P_;

  // set current laser NIS
  NIS_laser_ = CalculateNIS(S, z_diff);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  VectorXd z = meas_package.raw_measurements_;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_radar_, n_sigma_);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);

  // transform sigma points into measurement space
  for (int i = 0; i < n_sigma_; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double psi = Xsig_pred_(3, i);

    Zsig(0, i) = sqrt(px * px + py * py);
    if (sqrt(px * px + py * py) == 0) {
      Zsig(1, i) = 0;
    } else {
      Zsig(1, i) = atan2(py, px);
    }
    Zsig(2, i) = (px * cos(psi) * v + py * sin(psi) * v) / (sqrt(px * px + py * py));
  }

  // calculate mean predicted measurement
  z_pred = Zsig * weights_;

  // calculate measurement covariance matrix S
  for (int i = 0; i < n_sigma_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += (weights_[i] * z_diff * z_diff.transpose());
  }
  S += R_radar_;

  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);

  // calculate cross correlation matrix
  for (int i = 0; i < n_sigma_; i++) {
    VectorXd x_d = Xsig_pred_.col(i) - x_;
    VectorXd z_d = Zsig.col(i) - z_pred;
    Tc += weights_[i] * (x_d * z_d.transpose());
  }
  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = z - z_pred;

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  // set current radar NIS
  NIS_radar_ = CalculateNIS(S, z_diff);
}


void UKF::CreateAugmentedSigmaPoints() {
  // augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  // augmented state covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // augmented sigma points
  Xsig.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    VectorXd delta_x = sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig.col(1 + i) = x_aug + delta_x;
    Xsig.col(1 + n_aug_ + i) = x_aug - delta_x;
  }
}

void UKF::PredictAugmentedSigmaPoints(double delta_t) {
  // sigma point prediction
  for (int i = 0; i < n_sigma_; ++i) {
    double px = Xsig(0, i);
    double py = Xsig(1, i);
    double v = Xsig(2, i);
    double yaw = Xsig(3, i);
    double yaw_d = Xsig(4, i);
    double a = Xsig(5, i);
    double yaw_dd = Xsig(6, i);

    // with or without yaw speed
    if (fabs(yaw_d) < 1e-6) {
      Xsig_pred_(0, i) = px + v * cos(yaw) * delta_t;
      Xsig_pred_(1, i) = py + v * sin(yaw) * delta_t;
    } else {
      Xsig_pred_(0, i) = px + v / yaw_d * (sin(yaw + yaw_d * delta_t) - sin(yaw));
      Xsig_pred_(1, i) = py + v / yaw_d * (cos(yaw) - cos(yaw + yaw_d * delta_t));
    }
    // add noise
    double delta_t_2 = delta_t * delta_t;
    Xsig_pred_(0, i) += 0.5 * a * delta_t_2 * cos(yaw);
    Xsig_pred_(1, i) += 0.5 * a * delta_t_2 * sin(yaw);

    // speed, yaw, yaw_d, including noise
    Xsig_pred_(2, i) = v + a * delta_t;
    Xsig_pred_(3, i) = yaw + yaw_d * delta_t + 0.5 * yaw_dd * delta_t_2;
    Xsig_pred_(4, i) = yaw_d + delta_t * yaw_dd;
  }
}

double UKF::NormalizeAngle(double angle) {
  if (angle > M_PI || angle < -M_PI) {
    angle -= int(angle / M_PI) * M_PI;
  }
  return angle;
}

double UKF::CalculateNIS(const MatrixXd& S, const VectorXd& z_diff) {
  return z_diff.transpose() * S.inverse() * z_diff;
}