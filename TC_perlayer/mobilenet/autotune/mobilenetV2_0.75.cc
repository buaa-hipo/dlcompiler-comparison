#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
 
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <sys/time.h>

#include "common.h"

#include "tc/aten/aten.h"
#include "tc/aten/aten_autotuner.h"
#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/genetic_search.h"
#include "tc/core/check.h"
#include "tc/core/cpu/cpu_mapping_options.h"
#include "tc/core/cpu/cpu_tc_executor.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"

DEFINE_string(
    options_cache,
    "autotune_msg",
    "Filename to load and store best options cache"
);

std::string convolution_string = R"TC(
    def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1) -> (O) {
        O(n, m, h, w) +=! 
            I(n, r_c, h * <stride> + r_kh, w * <stride>+ r_kw) * W1(m, r_c, r_kh, r_kw)
    }
)TC";
std::string depthwise_string = R"TC(
    def convolution(float(N,C,H,W) I, float(C,KH,KW) K) -> (O) {
        O(n,c,h,w) +=!
            I(n, c, h * <stride> + r_kh, w * <stride> + r_kw) * K(c, r_kh, r_kw)
    }
)TC";
/**************************** 卷积层参数替换函数 ***************************/
std::string change_string_parameter(std::string kernel_string, int stride)
{
    while (true) {
        auto pos = kernel_string.find(std::string("<stride>"));
        if (pos == std::string::npos)
            break;
        kernel_string = kernel_string.replace(pos, std::string("<stride>").size(), std::to_string(stride));
    }

    return kernel_string;
}
/******************************* 最佳参数获取 ***********************************/
// 卷积层最佳参数获取
template <typename Backend>
std::vector<typename Backend::MappingOptionsType>
get_convolution_bestOption(std::string kernel_string, at::Tensor I0, at::Tensor I1)
{
    // 尝试从保存的文件中获取调优信息
    auto options = \
        tc::autotune::loadTopKFromCacheFile<Backend>( \
            kernel_string, \
            "convolution", \
            FLAGS_options_cache, \
            extractRawPtrs(tc::aten::makeDLConstTensors({I0, I1})), 1);

    if (options.size() == 0) {
        // 不存在调优信息, 从初始选项开始, 使用进化自动调优搜索
        auto naiveOptions = Backend::MappingOptionsType::makeNaiveMappingOptions();
        tc::aten::ATenAutotuner<Backend, tc::autotune::GeneticSearch> \
            geneticAutotuneATen(kernel_string);
        options = \
            geneticAutotuneATen.tune("convolution", {I0, I1}, {naiveOptions});
        // 保存相关最优信息
        if (not FLAGS_options_cache.empty()) {
            tc::autotune::appendTopKToCacheFile(
                *geneticAutotuneATen.optionsCache,
                FLAGS_options_cache,
                tc::FLAGS_tuner_save_best_candidates_count);
        }
    }

    return options;
}
/*********************** End of 最佳参数获取 ************************************/
/************************ 获取对应的kernel并得到输出 *****************************/
template <typename Backend>
double
get_convolution_output(std::string kernel_string, at::Tensor I0, at::Tensor I1, std::unique_ptr<typename Backend::ExecutorType> & pExecutor)
{
    auto outputs = tc::aten::prepareOutputs(kernel_string, "convolution", {I0, I1});
    tc::aten::profile(*pExecutor, {I0, I1}, outputs);
    auto timings = tc::aten::profile(*pExecutor, {I0, I1}, outputs);
    // std::cout << " GPU convolution ran in: " << timings.kernelRuntime.toMicroSeconds() << "us\n";

    return timings.kernelRuntime.toMicroSeconds();
}
/********************** End of 获取对应kernel并得到输出 **************************/
/********************************** GPU 测试 ***********************************/
TEST(mobilenet, 0_75)
{
    double time_use = 0;
    double time_use_sum = 0;
    
    std::string conv_S1_string = change_string_parameter(convolution_string, 1);
    std::string conv_S2_string = change_string_parameter(convolution_string, 2);
    std::string depthwise_S1_string = change_string_parameter(depthwise_string, 1);
    std::string depthwise_S2_string = change_string_parameter(depthwise_string, 2);
    // 申请随机数据tensor, 以获取对应kernel的最佳参数
    at::Tensor I0 = makeATenTensor<tc::CudaBackend>({1, 64, 58, 58});
    at::Tensor I1 = makeATenTensor<tc::CudaBackend>({64, 64, 3, 3});
    at::Tensor I2 = makeATenTensor<tc::CudaBackend>({64, 3, 3});
    auto bestOptions_conv_S1 = get_convolution_bestOption<tc::CudaBackend>(conv_S1_string, I0, I1);
    auto bestOptions_conv_S2 = get_convolution_bestOption<tc::CudaBackend>(conv_S2_string, I0, I1);
    auto bestOptions_depthwise_S1 = get_convolution_bestOption<tc::CudaBackend>(depthwise_S1_string, I0, I2);
    auto bestOptions_depthwise_S2 = get_convolution_bestOption<tc::CudaBackend>(depthwise_S2_string, I0, I2);
    // pointwise卷积本质上是常规卷积, 直接复用常规卷积最佳参数

    /******************* conv_1x3x224x224_24x3x3x3_S2P1 ************************/
    time_use_sum = 0;
    at::Tensor input = makeATenTensor<tc::CudaBackend>({1, 3, 226, 226}); 
    at::Tensor kernel = makeATenTensor<tc::CudaBackend>({24, 3, 3, 3});
    auto pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S2_string, "convolution", {input, kernel}, bestOptions_conv_S2[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x3x224x224_24x3x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x3x224x224_24x3x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x16x112x112_16x16x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 16, 112, 112}); 
    kernel = makeATenTensor<tc::CudaBackend>({16, 16, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x16x112x112_16x16x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x16x112x112_16x16x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x24x112x112_24x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 24, 114, 114}); 
    kernel = makeATenTensor<tc::CudaBackend>({24, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x24x112x112_24x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x24x112x112_24x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x24x112x112_12x24x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 24, 112, 112}); 
    kernel = makeATenTensor<tc::CudaBackend>({12, 24, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x24x112x112_12x24x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x24x112x112_12x24x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x12x112x112_72x12x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 12, 112, 112}); 
    kernel = makeATenTensor<tc::CudaBackend>({72, 12, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x12x112x112_72x12x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x12x112x112_72x12x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x72x112x112_72x3x3_S2P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 72, 114, 114}); 
    kernel = makeATenTensor<tc::CudaBackend>({72, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S2_string, "convolution", {input, kernel}, bestOptions_depthwise_S2[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x72x112x112_72x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x72x112x112_72x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x72x56x56_18x72x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 72, 56, 56}); 
    kernel = makeATenTensor<tc::CudaBackend>({18, 72, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x72x56x56_18x72x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x72x56x56_18x72x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x18x56x56_108x18x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 18, 56, 56}); 
    kernel = makeATenTensor<tc::CudaBackend>({108, 18, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x18x56x56_108x18x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x18x56x56_108x18x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x108x56x56_108x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 108, 58, 58}); 
    kernel = makeATenTensor<tc::CudaBackend>({108, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x108x56x56_108x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x108x56x56_108x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x108x56x56_18x108x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 108, 56, 56}); 
    kernel = makeATenTensor<tc::CudaBackend>({18, 108, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x108x56x56_18x108x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x108x56x56_18x108x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x108x56x56_108x3x3_S2P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 108, 58, 58}); 
    kernel = makeATenTensor<tc::CudaBackend>({108, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S2_string, "convolution", {input, kernel}, bestOptions_depthwise_S2[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x108x56x56_108x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x108x56x56_108x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x108x28x28_24x108x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 108, 28, 28}); 
    kernel = makeATenTensor<tc::CudaBackend>({24, 108, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x108x28x28_24x108x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x108x28x28_24x108x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x24x28x28_144x24x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 24, 28, 28}); 
    kernel = makeATenTensor<tc::CudaBackend>({144, 24, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x24x28x28_144x24x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x24x28x28_144x24x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x144x28x28_144x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 144, 30, 30}); 
    kernel = makeATenTensor<tc::CudaBackend>({144, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x144x28x28_144x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x144x28x28_144x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x144x28x28_24x144x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 144, 28, 28}); 
    kernel = makeATenTensor<tc::CudaBackend>({24, 144, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x144x28x28_24x144x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x144x28x28_24x144x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x144x28x28_144x3x3_S2P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 144, 30, 30}); 
    kernel = makeATenTensor<tc::CudaBackend>({144, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S2_string, "convolution", {input, kernel}, bestOptions_depthwise_S2[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x144x28x28_144x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x144x28x28_144x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x144x14x14_48x144x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 144, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({48, 144, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x144x14x14_48x144x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x144x14x14_48x144x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x48x14x14_288x48x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 48, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({288, 48, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x48x14x14_288x48x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x48x14x14_288x48x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x288x14x14_288x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 288, 16, 16}); 
    kernel = makeATenTensor<tc::CudaBackend>({288, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x288x14x14_288x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x288x14x14_288x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x288x14x14_48x288x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 288, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({48, 288, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x288x14x14_48x288x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x288x14x14_48x288x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x288x14x14_72x288x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 288, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({72, 288, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x288x14x14_72x288x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x288x14x14_72x288x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x72x14x14_432x72x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 72, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({432, 72, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x72x14x14_432x72x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x72x14x14_432x72x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x432x14x14_432x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 432, 16, 16}); 
    kernel = makeATenTensor<tc::CudaBackend>({432, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x432x14x14_432x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x432x14x14_432x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x432x14x14_72x432x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 432, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({72, 432, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x432x14x14_72x432x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x432x14x14_72x432x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x432x14x14_432x3x3_S2P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 432, 16, 16}); 
    kernel = makeATenTensor<tc::CudaBackend>({432, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S2_string, "convolution", {input, kernel}, bestOptions_depthwise_S2[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x432x14x14_432x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x432x14x14_432x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x432x7x7_120x432x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 432, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({120, 432, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x432x7x7_120x432x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x432x7x7_120x432x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x120x7x7_720x120x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 120, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({720, 120, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x120x7x7_720x120x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x120x7x7_720x120x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x720x7x7_720x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 720, 9, 9}); 
    kernel = makeATenTensor<tc::CudaBackend>({720, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x720x7x7_720x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x720x7x7_720x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x720x7x7_120x720x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 720, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({120, 720, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x720x7x7_120x720x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x720x7x7_120x720x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x720x7x7_240x720x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 720, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({240, 720, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x720x7x7_240x720x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x720x7x7_240x720x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x240x7x7_1280x240x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 240, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({1280, 240, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x240x7x7_1280x240x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x240x7x7_1280x240x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x1280x7x7_1000x1280x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 1280, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({1000, 1280, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1[0]);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x1280x7x7_1000x1280x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x1280x7x7_1000x1280x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

}
/***************************** End of GPU Test ********************************/
// From root, run with: ./build/tc/examples/mobile --tuner_threads=10 --tuner_gen_pop_size=10 --tuner_gen_generations=3 --tuner_gen_number_elites=4
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    ::google::InitGoogleLogging(argv[0]);
    tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
    return RUN_ALL_TESTS();
}