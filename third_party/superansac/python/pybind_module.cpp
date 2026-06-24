#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "inlier_selectors/types.h"
#include "local_optimization/types.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "settings.h"
#include "superansac.h"
#include "termination/types.h"

#include <array>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

std::tuple<Eigen::Matrix4d, std::vector<size_t>, double, size_t> estimateRigidTransform(
    const DataMatrix& correspondences,
    const std::vector<double>& bounding_box_sizes,
    const std::vector<double>& point_probabilities,
    superansac::RANSACSettings& settings);

static inline void require_2d(const py::buffer_info& b, const char* name) {
    if (b.ndim != 2) {
        throw std::runtime_error(std::string(name) + " must be a 2D array");
    }
}

template <int N>
static inline std::array<double, N> require_1d_fixed_f64(
    py::array_t<double, py::array::c_style> a,
    const char* name) {
    auto b = a.request();
    if (b.ndim != 1 || b.shape[0] != N) {
        throw std::runtime_error(
            std::string(name) + " must be a 1D float64 array of length " + std::to_string(N));
    }
    const double* p = static_cast<const double*>(b.ptr);
    std::array<double, N> out;
    for (int i = 0; i < N; ++i) {
        out[i] = p[i];
    }
    return out;
}

static inline std::vector<double> vec_from_1d_f64(
    py::array_t<double, py::array::c_style> a,
    const char* name) {
    auto b = a.request();
    if (b.ndim != 1) {
        throw std::runtime_error(std::string(name) + " must be a 1D float64 array");
    }
    const auto n = static_cast<size_t>(b.shape[0]);
    const double* p = static_cast<const double*>(b.ptr);
    return std::vector<double>(p, p + n);
}

static inline std::vector<double> probs_from_optional(py::object probabilities) {
    if (probabilities.is_none()) {
        return {};
    }
    auto a = py::cast<py::array_t<double, py::array::c_style>>(probabilities);
    return vec_from_1d_f64(a, "probabilities");
}

PYBIND11_MODULE(pysuperansac, m) {
    m.doc() = "SuperRANSAC rigid-transform bindings for SUFLECA";

    py::enum_<superansac::scoring::ScoringType>(m, "ScoringType")
        .value("RANSAC", superansac::scoring::ScoringType::RANSAC)
        .value("MAGSAC", superansac::scoring::ScoringType::MAGSAC)
        .export_values();

    py::enum_<superansac::samplers::SamplerType>(m, "SamplerType")
        .value("Uniform", superansac::samplers::SamplerType::Uniform)
        .value("PROSAC", superansac::samplers::SamplerType::PROSAC)
        .export_values();

    py::enum_<superansac::local_optimization::LocalOptimizationType>(m, "LocalOptimizationType")
        .value("Nothing", superansac::local_optimization::LocalOptimizationType::None)
        .export_values();

    py::enum_<superansac::termination::TerminationType>(m, "TerminationType")
        .value("RANSAC", superansac::termination::TerminationType::RANSAC)
        .export_values();

    py::enum_<superansac::inlier_selector::InlierSelectorType>(m, "InlierSelectorType")
        .value("Nothing", superansac::inlier_selector::InlierSelectorType::None)
        .export_values();

    py::class_<superansac::RANSACSettings>(m, "RANSACSettings")
        .def(py::init<>())
        .def_readwrite("min_iterations", &superansac::RANSACSettings::minIterations)
        .def_readwrite("max_iterations", &superansac::RANSACSettings::maxIterations)
        .def_readwrite("inlier_threshold", &superansac::RANSACSettings::inlierThreshold)
        .def_readwrite("confidence", &superansac::RANSACSettings::confidence)
        .def_readwrite("scoring", &superansac::RANSACSettings::scoring)
        .def_readwrite("sampler", &superansac::RANSACSettings::sampler)
        .def_readwrite("local_optimization", &superansac::RANSACSettings::localOptimization)
        .def_readwrite("final_optimization", &superansac::RANSACSettings::finalOptimization)
        .def_readwrite("termination_criterion", &superansac::RANSACSettings::terminationCriterion)
        .def_readwrite("inlier_selector", &superansac::RANSACSettings::inlierSelector)
        .def_readwrite("use_sprt", &superansac::RANSACSettings::useSprt);

    m.def(
        "estimateRigidTransform",
        [](py::array_t<double, py::array::c_style> correspondences,
           py::array_t<double, py::array::c_style> bounding_box_sizes,
           py::object probabilities,
           superansac::RANSACSettings& config) {
            auto buf = correspondences.request();
            require_2d(buf, "correspondences");

            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(
                static_cast<const double*>(buf.ptr),
                static_cast<Eigen::Index>(buf.shape[0]),
                static_cast<Eigen::Index>(buf.shape[1]));

            const auto bb6 = require_1d_fixed_f64<6>(bounding_box_sizes, "bounding_box_sizes");
            std::vector<double> bb_vec(bb6.begin(), bb6.end());
            std::vector<double> prob_vec = probs_from_optional(probabilities);

            py::gil_scoped_release release;
            return estimateRigidTransform(mat, bb_vec, prob_vec, config);
        },
        "Estimate a 3D-3D rigid transform from point correspondences.",
        py::arg("correspondences"),
        py::arg("bounding_box_sizes"),
        py::arg("probabilities"),
        py::arg("config"));
}
