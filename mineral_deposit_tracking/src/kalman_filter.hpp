#ifndef KALMAN_FILTER_HPP_

#define KALMAN_FILTER_HPP_
#include <Eigen/Dense>

namespace mineral_deposit_tracking
{

template<int StateSize>
class KalmanFilter
{
public:
	using VectorType = Eigen::Matrix<double, StateSize, 1>;
	using MatrixType = Eigen::Matrix<double, StateSize, StateSize>;

	KalmanFilter(
		const MatrixType & transition_matrix,
		const MatrixType & process_covariance,
		const MatrixType & observation_matrix)
	:	transition_matrix_(transition_matrix),
		process_covariance_(process_covariance),
		observation_matrix_(observation_matrix),
		estimate_(VectorType::Zero()),
		estimate_covariance_(MatrixType::Identity() * 500)
	{
	}

	void Reset(const VectorType & initial_state, const MatrixType & initial_covariance)
	{
		estimate_ = initial_state;
		estimate_covariance_ = initial_covariance;
	}

	void TimeUpdate()
	{
		estimate_ = transition_matrix_ * estimate_;
		estimate_covariance_ = (transition_matrix_ * estimate_covariance_ * transition_matrix_.transpose()) + process_covariance_;
	}

	void MeasurementUpdate(const VectorType & measurement, const MatrixType & measurement_covariance)
	{
		const MatrixType innovation_covariance = (observation_matrix_ * estimate_covariance_ * observation_matrix_.transpose()) + measurement_covariance;

		const MatrixType kalman_gain = estimate_covariance_ * observation_matrix_.transpose() * innovation_covariance.inverse();

		estimate_ = estimate_ + (kalman_gain * (measurement - (observation_matrix_ * estimate_)) );

		const MatrixType tmp = MatrixType::Identity() - (kalman_gain * observation_matrix_);

		estimate_covariance_ = ( tmp * estimate_covariance_ * tmp.transpose() ) + ( kalman_gain * measurement_covariance * kalman_gain.transpose() );

	}

	const VectorType & GetEstimate() const
	{
		return estimate_;
	}

	const MatrixType & GetEstimateCovariance() const
	{
		return estimate_covariance_;
	}

private:
	const MatrixType transition_matrix_;
	const MatrixType process_covariance_;
	const MatrixType observation_matrix_;
	VectorType estimate_;
	MatrixType estimate_covariance_;
};


}

#endif