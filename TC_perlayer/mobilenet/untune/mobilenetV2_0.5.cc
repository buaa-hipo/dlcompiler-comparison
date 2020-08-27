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
typename Backend::MappingOptionsType
get_convolution_bestOption(std::string kernel_string)
{
    auto options = Backend::MappingOptionsType::makeNaiveMappingOptions();
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
TEST(mobilenet, 0_5)
{
    double time_use = 0;
    double time_use_sum = 0;
    
    std::string conv_S1_string = change_string_parameter(convolution_string, 1);
    std::string conv_S2_string = change_string_parameter(convolution_string, 2);
    std::string depthwise_S1_string = change_string_parameter(depthwise_string, 1);
    std::string depthwise_S2_string = change_string_parameter(depthwise_string, 2);

    auto bestOptions_conv_S1 = get_convolution_bestOption<tc::CudaBackend>(conv_S1_string);
    auto bestOptions_conv_S2 = get_convolution_bestOption<tc::CudaBackend>(conv_S2_string);
    auto bestOptions_depthwise_S1 = get_convolution_bestOption<tc::CudaBackend>(depthwise_S1_string);
    auto bestOptions_depthwise_S2 = get_convolution_bestOption<tc::CudaBackend>(depthwise_S2_string);
    // pointwise卷积本质上是常规卷积, 直接复用常规卷积最佳参数

    /******************* conv_1x3x224x224_16x3x3x3_S2P1 ************************/
    time_use_sum = 0;
    at::Tensor input = makeATenTensor<tc::CudaBackend>({1, 3, 226, 226}); 
    at::Tensor kernel = makeATenTensor<tc::CudaBackend>({16, 3, 3, 3});
    auto pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S2_string, "convolution", {input, kernel}, bestOptions_conv_S2);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x3x224x224_16x3x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x3x224x224_16x3x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x16x112x112_16x16x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 16, 112, 112}); 
    kernel = makeATenTensor<tc::CudaBackend>({16, 16, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x16x112x112_16x16x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x16x112x112_16x16x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x16x112x112_16x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 16, 114, 114}); 
    kernel = makeATenTensor<tc::CudaBackend>({16, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x16x112x112_16x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x16x112x112_16x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x16x112x112_8x16x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 16, 112, 112}); 
    kernel = makeATenTensor<tc::CudaBackend>({8, 16, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x16x112x112_8x16x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x16x112x112_8x16x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x8x112x112_48x8x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 8, 112, 112}); 
    kernel = makeATenTensor<tc::CudaBackend>({48, 8, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x8x112x112_48x8x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x8x112x112_48x8x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x48x112x112_48x3x3_S2P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 48, 114, 114}); 
    kernel = makeATenTensor<tc::CudaBackend>({48, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S2_string, "convolution", {input, kernel}, bestOptions_depthwise_S2);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x48x112x112_48x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x48x112x112_48x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x48x56x56_12x48x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 48, 56, 56}); 
    kernel = makeATenTensor<tc::CudaBackend>({12, 48, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x48x56x56_12x48x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x48x56x56_12x48x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x12x56x56_72x12x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 12, 56, 56}); 
    kernel = makeATenTensor<tc::CudaBackend>({72, 12, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x12x56x56_72x12x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x12x56x56_72x12x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x72x56x56_72x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 72, 58, 58}); 
    kernel = makeATenTensor<tc::CudaBackend>({72, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x72x56x56_72x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x72x56x56_72x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x72x56x56_12x72x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 72, 56, 56}); 
    kernel = makeATenTensor<tc::CudaBackend>({12, 72, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x72x56x56_12x72x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x72x56x56_12x72x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x72x56x56_72x3x3_S2P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 72, 58, 58}); 
    kernel = makeATenTensor<tc::CudaBackend>({72, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S2_string, "convolution", {input, kernel}, bestOptions_depthwise_S2);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x72x56x56_72x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x72x56x56_72x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x72x28x28_16x72x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 72, 28, 28}); 
    kernel = makeATenTensor<tc::CudaBackend>({16, 72, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x72x28x28_16x72x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x72x28x28_16x72x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x16x28x28_96x16x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 16, 28, 28}); 
    kernel = makeATenTensor<tc::CudaBackend>({96, 16, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x16x28x28_96x16x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x16x28x28_96x16x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x96x28x28_96x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 96, 30, 30}); 
    kernel = makeATenTensor<tc::CudaBackend>({96, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x96x28x28_96x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x96x28x28_96x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x96x28x28_16x96x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 96, 28, 28}); 
    kernel = makeATenTensor<tc::CudaBackend>({16, 96, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x96x28x28_16x96x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x96x28x28_16x96x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x96x28x28_96x3x3_S2P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 96, 30, 30}); 
    kernel = makeATenTensor<tc::CudaBackend>({96, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S2_string, "convolution", {input, kernel}, bestOptions_depthwise_S2);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x96x28x28_96x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x96x28x28_96x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x96x14x14_32x96x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 96, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({32, 96, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x96x14x14_32x96x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x96x14x14_32x96x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x32x14x14_192x32x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 32, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({192, 32, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x32x14x14_192x32x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x32x14x14_192x32x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x192x14x14_192x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 96, 16, 16}); 
    kernel = makeATenTensor<tc::CudaBackend>({96, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x192x14x14_192x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x192x14x14_192x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x192x14x14_32x192x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 192, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({32, 192, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x192x14x14_32x192x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x192x14x14_32x192x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x192x14x14_48x192x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 192, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({48, 192, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x192x14x14_48x192x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x192x14x14_48x192x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x48x14x14_288x48x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 48, 14, 14}); 
    kernel = makeATenTensor<tc::CudaBackend>({288, 48, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
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
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1);
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
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x288x14x14_48x288x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x288x14x14_48x288x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x288x14x14_288x3x3_S2P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 288, 16, 16}); 
    kernel = makeATenTensor<tc::CudaBackend>({288, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S2_string, "convolution", {input, kernel}, bestOptions_depthwise_S2);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S2_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x288x14x14_288x3x3_S2P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x288x14x14_288x3x3_S2P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x288x7x7_80x288x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 288, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({80, 288, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x288x7x7_80x288x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x288x7x7_80x288x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x80x7x7_480x80x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 80, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({480, 80, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x80x7x7_480x80x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x80x7x7_480x80x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* depthwise_1x480x7x7_480x3x3_S1P1 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 480, 9, 9}); 
    kernel = makeATenTensor<tc::CudaBackend>({480, 3, 3});
    pExecutor = tc::aten::compile<tc::CudaBackend>(depthwise_S1_string, "convolution", {input, kernel}, bestOptions_depthwise_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(depthwise_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/depthwise_1x480x7x7_480x3x3_S1P1[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/depthwise_1x480x7x7_480x3x3_S1P1[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x480x7x7_80x480x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 480, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({80, 480, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x480x7x7_80x480x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x480x7x7_80x480x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x480x7x7_160x480x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 480, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({160, 480, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x480x7x7_160x480x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x480x7x7_160x480x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x160x7x7_1280x160x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 160, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({1280, 160, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
    for (int i = 0; i < 15; i++) {

        time_use = get_convolution_output<tc::CudaBackend>(conv_S1_string, input, kernel, pExecutor);

        if (i >= 5) time_use_sum += time_use;
        std::cout << "mobilenet/conv_1x160x7x7_1280x160x1x1_S1P0[Round" << i <<"]: " << time_use << " us" << std::endl;
    }
	time_use_sum /= 10;
	std::cout << "mobilenet/conv_1x160x7x7_1280x160x1x1_S1P0[Time]: " << time_use_sum << " us" << std::endl;

    /******************* conv_1x1280x7x7_1000x1280x1x1_S1P0 ************************/
    time_use_sum = 0;
    input = makeATenTensor<tc::CudaBackend>({1, 1280, 7, 7}); 
    kernel = makeATenTensor<tc::CudaBackend>({1000, 1280, 1, 1});
    pExecutor = tc::aten::compile<tc::CudaBackend>(conv_S1_string, "convolution", {input, kernel}, bestOptions_conv_S1);
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