---
layout: post
title:  Training a TensorFlow graph in C++ API
date:   2016-07-14 15:42:13
categories:
tags:
permalink: /posts/Training-a-TensorFlow-graph-in-C++-API
published: true
---

First off, I want to explain my motivation for training the model in C++ and why you may want to do this. TensorFlow is written in C/C++ wrapped with SWIG to obtain python bindings providing speed and usability. However, when a call from python is made to C/C++ e.g. TensorFlow or numpy. Python's global interpreter lock (GIL) must be acquired to perform each call. A few context switches are fine but repeated calls can gradually add up such as performing a true stochastic gradient descent. Moreover, integrating other models with deep learning that do not run effectively on GPUs can require a lot of costly memory transfers. To avoid this I decided to run it directly in C++ providing better performance and finer grain control of GPU memory allocations.


In this tutorial, I assume you are able to build TensorFlow from source. If not instructions can be found on the TensorFlow website [here](https://www.tensorflow.org/versions/master/get_started/os_setup.html#installing-from-sources). It will be also useful if you have some experience in [adding a new op](https://www.tensorflow.org/versions/master/how_tos/adding_an_op/index.html#adding-a-new-op) although not necessary. I highly recommend going through this tutorial on [Loading a TensorFlow graph with the C++ API](https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.d0stu13xd) since I am merely extending this.


The code and graph can be found in my [repository](https://github.com/tebesu/Tensorflow-Cpp-API-Training).


<div style="margin-top:25px"></div>

## Building the Graph
First we need to create a graph, it is not worth the effort to construct it in C++ here is an [example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/tutorials/example_trainer.cc#L49) to prove my point. In python, create a graph definition file and export it as a binary protobuf. It is crucial to name the variables and operations as we will be calling the function by names. For simplicity we will use a simple feedforward neural network.

{% highlight python linenos %}
with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, 32], name="x")
    y = tf.placeholder(tf.float32, [None, 8], name="y")

    w1 = tf.Variable(tf.truncated_normal([32, 16], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.0, shape=[16]))

    w2 = tf.Variable(tf.truncated_normal([16, 8], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.0, shape=[8]))

    a = tf.nn.tanh(tf.nn.bias_add(tf.matmul(x, w1), b1))
    y_out = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a, w2), b2), name="y_out")
    cost = tf.reduce_sum(tf.square(y-y_out), name="cost")
    optimizer = tf.train.AdamOptimizer().minimize(cost, name="train")

    init = tf.initialize_variables(tf.all_variables(), name='init_all_vars_op')
    tf.train.write_graph(sess.graph_def,
                         './',
                         'mlp.pb', as_text=False)
                         {% endhighlight %}


<div style="margin-top:75px"></div>


## Running in C++


To run a graph in C++ the steps are loading the graph definition; creating a session; initialize the graph within the session and running our graph. I will first present the entire code then provide explanations.


{% highlight c++ linenos %}
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
using namespace tensorflow;

int main(int argc, char* argv[]) {

    std::string graph_definition = "mlp.pb";
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    std::vector<Tensor> outputs; // Store outputs
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));

    // Set GPU options
    graph::SetDefaultDevice("/gpu:0", &graph_def);
    opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
    opts.config.mutable_gpu_options()->set_allow_growth(true);

    // create a new session
    TF_CHECK_OK(NewSession(opts, &session));

    // Load graph into session
    TF_CHECK_OK(session->Create(graph_def));

    // Initialize our variables
    TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));

    Tensor x(DT_FLOAT, TensorShape({100, 32}));
    Tensor y(DT_FLOAT, TensorShape({100, 8}));
    auto _XTensor = x.matrix<float>();
    auto _YTensor = y.matrix<float>();

    _XTensor.setRandom();
    _YTensor.setRandom();

    for (int i = 0; i < 10; ++i) {
{% raw %}
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"cost"}, {}, &outputs)); // Get cost
        float cost = outputs[0].scalar<float>()(0);
        std::cout << "Cost: " <<  cost << std::endl;
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr)); // Train
        outputs.clear();{% endraw %}
    }


    session->Close();
    delete session;
    return 0;
}
{% endhighlight %}

<div style="margin-top:50px"></div>

If we examine the code we will see many similarities with running the Python code.
Lines 15-17 set GPU options for the graph definition. We set the default device as "/gpu:0", set the memory fraction as 0.5 and allow growth on the GPU memory usage. These settings are the same when initializing a session with a tf.config in python.

Lines 20 create a new session with the options we specified. Line 23 loads the graph definition into the session so we can use it. Line 26 we initialize all our variables like we would in any tensorflow session.

Lines 28-31 we initialize a tensor as our inputs and outputs.
TensorFlow’s tensors are underlying Eigen tensors. Essentially, we call x.matrix<float>() to get a pointer to Eigen’s tensor and hence the underlying data. We can similarly call x.vector, x.scalar and so on... See Eigen's Tensor Documentation and TensorFlow's Tensor Documentation for more details. Lines 33-34 generates some random data.


Lines 36-43 is where the real computation happens. Recall in our graph definition we explicitly named some variables and operations. Here we reference them by name as a string and providing the necessary inputs. The output is obtained by passing a vector that is populated when the graph is run (Lines 38-40). Lastly, Line 41 performs the training for the neural network. The remaining lines close the session and clean up our pointer.


### Compiling
There are two ways to compile this: one is bazel and the other is linking against the tensorflow library. I prefer the latter. For bazel see this post [Loading a TensorFlow graph with the C++ API](https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.d0stu13xd). To link against tensorflow first build the library via

{% highlight bash %}
bazel build -c opt --config=cuda //tensorflow:libtensorflow_cc.so
{% endhighlight %}

then add the proper flags to your compiler. After compiling and running you should get output similar to below:

{% highlight bash %}
Cost: 280.796
Cost: 272.988
Cost: 265.294
Cost: 257.712
Cost: 250.244
Cost: 242.897
Cost: 235.68
Cost: 228.594
Cost: 221.642
Cost: 214.822
{% endhighlight %}
