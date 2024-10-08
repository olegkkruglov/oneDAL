/* file: gbt_classification_predict_kernel.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Declaration of template function that computes gradient boosted trees
//  prediction results.
//--
*/

#ifndef __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "algorithms/gradient_boosted_trees/gbt_classification_predict.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace prediction
{
namespace internal
{
template <typename algorithmFpType, gbt::classification::prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute gradient boosted trees prediction results.
     *
     *  \param a[in]            Matrix of input variables X
     *  \param m[in]            Gradient boosted trees model obtained on training stage
     *  \param r[out]           Prediction results
     *  \param prob[out]        Prediction class probabilities
     *  \param nClasses[in]     Number of classes in gradient boosted trees algorithm parameter
     *  \param nIterations[in]  Number of iterations to predict in gradient boosted trees algorithm parameter
     *  \param predShapContributions[in] Predict SHAP contributions
     *  \param predShapInteractions[in] Predict SHAP interactions
     */
    services::Status compute(services::HostAppIface * pHostApp, const NumericTable * a, const classification::Model * m, NumericTable * r,
                             NumericTable * prob, size_t nClasses, size_t nIterations, bool predShapContributions, bool predShapInteractions);
};

} // namespace internal
} // namespace prediction
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
