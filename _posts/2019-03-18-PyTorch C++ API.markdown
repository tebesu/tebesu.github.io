---
layout: post
title:  PyTorch C++ Frontend Tutorial
date:   2019-03-18
categories: []
tags: [pytorch, c++]
permalink: /posts/PyTorch-C++-Frontend
published: true
summary:    How to define, train and run a PyTorch model using the C++ frontend API.
---




In this blog post, I will demonstrate how to define a model and train it in the PyTorch C++ API front end. Since not everyone has access to a DGX-2 to train their [Progressive GAN](https://arxiv.org/abs/1710.10196) in one week. I looked for ways to speed up the training of the model. Naturally changing to a lower level language should provide some speed ups.  Unfortunately (or fortunately), deep learning models are compute bound so overhead from acquiring the Python GIL is negligible. My timings show only around a 3% speed up. I'm sure more speed up may be obtained if high levels of preprocessing/IO operations are required or highly optimized C++ code. TLDR; just show me the [code repo](https://github.com/tebesu/pytorch-cpp-tutorial).

I will not cover the installation/setup for the PyTorch C++ API Front End. Please refer to the official documentation [here](https://pytorch.org/cppdocs/installing.html) and a basic tutorial provided on the PyTorch website [here](https://pytorch.org/tutorials/advanced/cpp_frontend.html).


Lets get started by implementing AlexNet as our example. I followed the existing Python implementation of AlexNet in [torchvision ](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py). Similar to the Python version we define a nn.Module with all of the required layers.



{% highlight c++ linenos %}
#include <iostream>
#include <vector>

#include <torch/torch.h>
using namespace torch;
using namespace std;

struct AlexNetImpl : torch::nn::Module {

    AlexNetImpl(int64_t N, int64_t M)
            : conv1(register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)))),
            conv2(register_module("conv2", nn::Conv2d(nn::Conv2dOptions(64, 192, 5).padding(2)))),
            conv3(register_module("conv3", nn::Conv2d(nn::Conv2dOptions(192, 384, 3).padding(1)))),
            conv4(register_module("conv4", nn::Conv2d(nn::Conv2dOptions(384, 256, 3).padding(1)))),
            conv5(register_module("conv5", nn::Conv2d(nn::Conv2dOptions(256, 256, 3).padding(1)))),
            linear1(register_module("linear1", nn::Linear(256*6*6, 4096))),
            linear2(register_module("linear2", nn::Linear(4096, 4096))),
            linear3(register_module("linear3", nn::Linear(4096, 1000))),
            dropout(register_module("dropout", nn::Dropout(nn::DropoutOptions(0.5)))){}
{% endhighlight %}

<br>

Initializing nn modules is a little bit more involved in the C++ version since we need to explicitly register each module rather than Python doing it for us. We defined 5 convolution layers, 3 fully connected layers and a dropout layer in the model constructor. Next, lets define the forward pass which follows line by line to the Python version.


{% highlight c++ linenos %}
    torch::Tensor forward(const torch::Tensor& input) {
        auto x = torch::relu(conv1(input));
        x = torch::max_pool2d(x, 3, 2);

        x = relu(conv2(x));
        x = max_pool2d(x, 3, 2);

        x = relu(conv3(x));
        x = relu(conv4(x));
        x = relu(conv5(x));
        x = max_pool2d(x, 3, 2);

        x = x.view({x.size(0), 256 * 6 * 6});
        x = dropout(x);
        x = relu(linear1(x));

        x = dropout(x);
        x = relu(linear2(x));

        x = linear3(x);
        return x;
    }

    // Module Layers
    torch::nn::Linear linear1, linear2, linear3;
    nn::Dropout dropout;
    nn::Conv2d conv1, conv2, conv3, conv4, conv5;
};
{% endhighlight %}

<br>

Next lets wrap the implementation following the standard convention of [Module Ownership](https://pytorch.org/tutorials/advanced/cpp_frontend.html#module-ownership) and the definition can be found here [TORCH_MODULE_IMPL](https://github.com/pytorch/pytorch/blob/4bdaca827cc7b71b33210c0ed4f202540d6719f7/torch/csrc/api/include/torch/nn/pimpl.h#L195). Basically, this wraps our implementation of AlexNetImpl to AlexNet with a shared_ptr and abstracts away any memory management.

<div class="small-spacer"></div>

{% highlight c++ %}
TORCH_MODULE_IMPL(AlexNet, AlexNetImpl);
{% endhighlight %}


<br>
Now that we finished defining the model we can start on the driver program. First lets check if any CUDA devices are available and set it as our default if possible (otherwise it will run on the CPU).
<div class="small-spacer"></div>

{% highlight c++ linenos %}
torch::Device device = torch::kCPU;
std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
}
{% endhighlight %}

<br>
Next lets initialize our model and an Adam optimizer.

{% highlight c++ linenos %}
int batch_size = 128;
int iterations = 1000;
auto model = AlexNet(224);
torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(1e-3));
{% endhighlight %}

<br>
Set our model to be in training mode (for dropout) and transfer it to the selected device (GPU if available).

{% highlight c++ linenos %}
model->train();
model->to(device);
{% endhighlight %}

<br>

Now the main training loop, we generate random normal tensors as both the inputs (x) and labels (targets) for the model with a simple mean squared error loss. This is just an example we are not doing anything particularly useful. As in a traditional PyTorch training loop, we zero out the gradients, compute the loss and update the parameters (step) with the optimizer.

{% highlight c++ linenos %}
torch::Tensor x, target, y, loss;

target = torch::randn({batch_size, 1000}, device);
x = torch::ones({batch_size, 3, 224, 224}, device);

for (int i = 0; i < iterations; ++i) {
    optim.zero_grad();
    y = model->forward(x);
    loss = torch::mse_loss(y, target);
    loss.backward();
    optim.step();
    if(i%10 == 0)
      cout << loss << endl;
}
{% endhighlight %}

<br>

Finally our cmake file to compile the file and link it with torchlib.

{% highlight cmake %}
cmake_minimum_required(VERSION 3.5)
project(torchtest)
set(CMAKE_BUILD_TYPE  Release)

find_package(Torch REQUIRED)

add_executable(train main.cpp)
target_link_libraries(train "${TORCH_LIBRARIES}")
{% endhighlight %}

<br>
At the command line, (asssuming you have libtorch in the same directory) we can run cmake, compile and run.
{% highlight bash %}
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=`pwd`/libtorch .. && make
./train
{% endhighlight %}

<br>
Heres the output.

{% highlight bash %}
CUDA DEVICE COUNT: 4
CUDA is available! Training on GPU.
1.00247
[ Variable[CUDAFloatType]{} ]
1.00029
[ Variable[CUDAFloatType]{} ]
1.00081
[ Variable[CUDAFloatType]{} ]
0.997911
[ Variable[CUDAFloatType]{} ]
0.99561
[ Variable[CUDAFloatType]{} ]
{% endhighlight %}


<br>

The full code can be found in the github repository [here](https://github.com/tebesu/pytorch-cpp-tutorial).
Note: I also included a image dataset loader with OpenCV as FilenameDataset in the repo.


<br>
