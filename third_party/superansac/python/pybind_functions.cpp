#include <Eigen/Dense>

#include "estimators/estimator_rigid_transformation.h"
#include "estimators/solver_rigid_transform_proscrutes.h"
#include "models/model.h"
#include "samplers/prosac_sampler.h"
#include "samplers/types.h"
#include "scoring/magsac_scoring.h"
#include "scoring/magsac_sprt_scoring.h"
#include "scoring/types.h"
#include "superansac.h"
#include "termination/ransac_criterion.h"
#include "termination/types.h"
#include "utils/types.h"

#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

std::tuple<Eigen::Matrix4d, std::vector<size_t>, double, size_t> estimateRigidTransform(
    const DataMatrix& correspondences,
    const std::vector<double>& bounding_box_sizes,
    const std::vector<double>& point_probabilities,
    superansac::RANSACSettings& settings)
{
    (void)bounding_box_sizes;
    if (correspondences.cols() != 6) {
        throw std::invalid_argument("The input matrix must have 6 columns (x1, y1, z1, x2, y2, z2).");
    }
    if (correspondences.rows() < 3) {
        throw std::invalid_argument("The input matrix must have at least 3 rows.");
    }
    (void)point_probabilities;
    if (settings.localOptimization != superansac::local_optimization::LocalOptimizationType::None ||
        settings.finalOptimization != superansac::local_optimization::LocalOptimizationType::None) {
        throw std::invalid_argument("Local optimization is not enabled in this build.");
    }
    if (settings.inlierSelector != superansac::inlier_selector::InlierSelectorType::None) {
        throw std::invalid_argument("Inlier selectors are not enabled in this build.");
    }

    auto estimator = std::make_unique<superansac::estimator::RigidTransformationEstimator>();
    estimator->setMinimalSolver(new superansac::estimator::solver::RigidTransformProscrutesSolver());
    estimator->setNonMinimalSolver(new superansac::estimator::solver::RigidTransformProscrutesSolver());

    std::unique_ptr<superansac::samplers::AbstractSampler> sampler =
        superansac::samplers::createSampler<6>(settings.sampler);
    if (settings.sampler == superansac::samplers::SamplerType::PROSAC) {
        dynamic_cast<superansac::samplers::PROSACSampler*>(sampler.get())->setSampleSize(estimator->sampleSize());
    }
    sampler->initialize(correspondences);

    std::unique_ptr<superansac::scoring::AbstractScoring> scorer =
        superansac::scoring::createScoring<6>(settings.scoring, settings.useSprt);
    scorer->setThreshold(settings.inlierThreshold);
    if (settings.scoring == superansac::scoring::ScoringType::MAGSAC) {
        if (settings.useSprt) {
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring*>(scorer.get())->initialize(estimator.get());
        } else {
            dynamic_cast<superansac::scoring::MAGSACScoring*>(scorer.get())->initialize(estimator.get());
        }
    }

    std::unique_ptr<superansac::termination::AbstractCriterion> termination =
        superansac::termination::createTerminationCriterion(settings.terminationCriterion);
    if (settings.terminationCriterion == superansac::termination::TerminationType::RANSAC) {
        dynamic_cast<superansac::termination::RANSACCriterion*>(termination.get())->setConfidence(settings.confidence);
    }

    superansac::SupeRansac robust_estimator;
    robust_estimator.setEstimator(estimator.get());
    robust_estimator.setSampler(sampler.get());
    robust_estimator.setScoring(scorer.get());
    robust_estimator.setTerminationCriterion(termination.get());
    robust_estimator.setSettings(settings);
    robust_estimator.run(correspondences);

    if (robust_estimator.getInliers().size() < estimator->sampleSize()) {
        return std::make_tuple(
            Eigen::Matrix4d::Identity(),
            std::vector<size_t>(),
            0.0,
            robust_estimator.getIterationNumber());
    }

    return std::make_tuple(
        robust_estimator.getBestModel().getData(),
        robust_estimator.getInliers(),
        robust_estimator.getBestScore().getValue(),
        robust_estimator.getIterationNumber());
}
