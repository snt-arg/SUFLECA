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
//     * Neither the name of ETH Zurich nor the
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
// Author: Daniel Barath (majti89@gmail.com)
#pragma once

#include "scoring/types.h"
#include "samplers/types.h"
#include "local_optimization/types.h"
#include "termination/types.h"
#include "inlier_selectors/types.h"

namespace superansac {
    struct RANSACSettings
    {
        size_t numSamples = 3;
        size_t topKForLocalOptimization = 3, // Number of best models used for local optimization (reduced from 5)
            minIterations = 1000, // Minimum number of iterations
            maxIterations = 5000; // Maximum number of iterations
        double inlierThreshold = 1.5; // Inlier threshold
        double confidence = 0.99; // Confidence
        bool localOptimizationInsideTheLoop = false, // Whether to locally optimize models when you find a new best or later
            useSprt = true; // Whether to use SPRT test to speed up
        
        scoring::ScoringType scoring = 
            scoring::ScoringType::MAGSAC; // Scoring type

        samplers::SamplerType sampler = 
            samplers::SamplerType::PROSAC; // Sampler type
            
        local_optimization::LocalOptimizationType localOptimization = 
            local_optimization::LocalOptimizationType::None; // Local optimization type

        local_optimization::LocalOptimizationType finalOptimization = 
            local_optimization::LocalOptimizationType::None; // Final optimization type

        termination::TerminationType terminationCriterion = 
            termination::TerminationType::RANSAC; // Termination criterion type

        inlier_selector::InlierSelectorType inlierSelector = 
            inlier_selector::InlierSelectorType::None; // Inlier selector type
    };

}
