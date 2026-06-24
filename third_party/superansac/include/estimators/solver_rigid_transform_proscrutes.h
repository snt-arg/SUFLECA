// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <cmath>
#include <unsupported/Eigen/Polynomials>

#include "abstract_solver.h"

namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class RigidTransformProscrutesSolver : public AbstractSolver
			{
			public:
				RigidTransformProscrutesSolver()
				{
				}

				~RigidTransformProscrutesSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				bool returnMultipleModels() const override
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				size_t maximumSolutions() const override
				{
					return 1;
				}
				
				// The minimum number of points required for the estimation
				size_t sampleSize() const override
				{		
					// for robustness, we use more than true minimal
					return 7;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				FORCE_INLINE bool estimateModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_ = nullptr) const override; // The weight for each point
                    

			protected:
				FORCE_INLINE bool estimateMinimalModel(
					const DataMatrix& kData_, // The set of data points
					const size_t *kSample_, // The sample used for the estimation
					const size_t kSampleNumber_, // The size of the sample
					std::vector<models::Model> &models_, // The estimated model parameters
					const double *kWeights_) const; // The weight for each point
			};

			FORCE_INLINE bool RigidTransformProscrutesSolver::estimateModel(
				const DataMatrix& kData_,
				const size_t *kSample_,
				const size_t kSampleNumber_,
				std::vector<models::Model> &models_,
				const double *kWeights_) const
			{
				(void)kWeights_;
				if (kSampleNumber_ < sampleSize())
					return false;

				constexpr double kEps = 1e-12;

				// Centroids
				Eigen::Vector3d centroidA = Eigen::Vector3d::Zero();
				Eigen::Vector3d centroidB = Eigen::Vector3d::Zero();

				for (size_t i = 0; i < kSampleNumber_; i++)
				{
					const size_t idx = kSample_ ? kSample_[i] : i;
					if (idx >= static_cast<size_t>(kData_.rows()))
						return false;

					centroidB += Eigen::Vector3d(
						kData_(idx,0),
						kData_(idx,1),
						kData_(idx,2));

					centroidA += Eigen::Vector3d(
						kData_(idx,3),
						kData_(idx,4),
						kData_(idx,5));
				}

				centroidA /= kSampleNumber_;
				centroidB /= kSampleNumber_;

				// Centered matrices
				Eigen::MatrixXd A(3, kSampleNumber_);
				Eigen::MatrixXd B(3, kSampleNumber_);

				for (size_t i = 0; i < kSampleNumber_; i++)
				{
					const size_t idx = kSample_ ? kSample_[i] : i;

					Eigen::Vector3d pB(
						kData_(idx,0),
						kData_(idx,1),
						kData_(idx,2));

					Eigen::Vector3d pA(
						kData_(idx,3),
						kData_(idx,4),
						kData_(idx,5));

					A.col(i) = pA - centroidA;
					B.col(i) = pB - centroidB;
				}

				// H = A_center * B_center'
				Eigen::Matrix3d H = A * B.transpose();

				// P = B_center * B_center'
				Eigen::Matrix3d P = B * B.transpose();

				if (!H.allFinite() || !P.allFinite())
					return false;

				Eigen::FullPivLU<Eigen::Matrix3d> luP(P);
				if (!luP.isInvertible())
					return false;

				// K_d = H_d * P_d^{-1}
				Eigen::Matrix3d Kd = H * luP.inverse();
				if (!Kd.allFinite())
					return false;

				// SVD(K_d)
				Eigen::JacobiSVD<Eigen::Matrix3d> svdK(
					Kd, Eigen::ComputeFullU | Eigen::ComputeFullV);

				Eigen::Matrix3d U = svdK.matrixU();
				Eigen::Matrix3d V = svdK.matrixV();

				// Rotation
				Eigen::Matrix3d R = U * V.transpose();

				// Ensure proper rotation (no reflection)
				if (R.determinant() < 0)
				{
					Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
					S(2,2) = -1;
					R = U * S * V.transpose();
				}
				if (!R.allFinite())
					return false;

				// Compute anisotropic scales
				double s1 = (A.transpose() * R * Eigen::DiagonalMatrix<double,3>(1,0,0) * B).trace() /
							(B.transpose() * Eigen::DiagonalMatrix<double,3>(1,0,0) * B).trace();

				double s2 = (A.transpose() * R * Eigen::DiagonalMatrix<double,3>(0,1,0) * B).trace() /
							(B.transpose() * Eigen::DiagonalMatrix<double,3>(0,1,0) * B).trace();

				double s3 = (A.transpose() * R * Eigen::DiagonalMatrix<double,3>(0,0,1) * B).trace() /
							(B.transpose() * Eigen::DiagonalMatrix<double,3>(0,0,1) * B).trace();
			
				if (!std::isfinite(s1) || !std::isfinite(s2) || !std::isfinite(s3))
					return false;

				Eigen::Matrix3d Scale = Eigen::Matrix3d::Zero();
				Scale(0,0) = s1;
				Scale(1,1) = s2;
				Scale(2,2) = s3;

				// Translation
				Eigen::Vector3d t = centroidA - R * Scale * centroidB;
				if (!t.allFinite())
					return false;

				// Build model
				models::Model model;
				auto &modelData = model.getMutableData();
				modelData.resize(4,4);

				Eigen::Matrix3d RS = R * Scale;
				if (!RS.allFinite())
					return false;

				modelData <<
					RS(0,0), RS(1,0), RS(2,0), 0,
					RS(0,1), RS(1,1), RS(2,1), 0,
					RS(0,2), RS(1,2), RS(2,2), 0,
					t(0),    t(1),    t(2),    1;

				models_.push_back(model);
				return true;
			}
		}
	}
}