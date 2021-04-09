# Increasing performance and reducing cost of deep learning inference using Amazon SageMaker Neo and Amazon Elastic Inference

When running deep learning models in production, balancing infrastructure cost
versus model latency is always an important consideration for AWS customers. At
re:Invent 2018, AWS introduced [Amazon SageMaker Neo](https://aws.amazon.com/sagemaker/neo/) and [Amazon Elastic
Inference](https://aws.amazon.com/machine-learning/elastic-inference/), two services that can make models more efficient for deep learning.
Elastic Inference is a hardware solution that provides the optimal amount of GPU
compute to perform inference. Neo is a software solution that optimizes deep
learning models for specific infrastructure deployments by reducing memory
imprint, which enables up to double the execution speed.
This post deploys an MXNet hot dog / not hot dog image classification model in
Amazon SageMaker to measure model latency and costs using Neo and Elastic
Inference in a variety of deployment scenarios. This post evaluates deployment
options using Amazon SageMaker and the different results you may see if you use
Amazon [EC2](http://aws.amazon.com/ec2) instances.

## The benefits of Elastic Inference
Elastic Inference allows you to allocate the right amount of GPU for compute as
needed instead of over-provisioning hardware for memory and CPU components by
using a regular p2/p3 instance. This efficiency can reduce deep learning inference
costs by up to 75%. Elastic Inference also features three separate sizes of GPU
acceleration (eia2.medium, eia2.large, and eia2.xlarge), which creates granularity
for optimizing cost and performance for a variety of use cases, from natural
language processing to computer vision. Similar to standard Amazon SageMaker
endpoints, you can easily scale Elastic Inference accelerators (EIA) by using [Amazon
EC2 Auto Scaling groups](http://aws.amazon.com/ec2/autoscaling), which allows you to scale your compute demands

## The benefits of Neo
Neo uses deep learning to find code optimizations for specific hardware and deep
learning frameworks that allow models to perform up to twice the speed with no
loss in accuracy. Furthermore, by reducing the code base for deep learning networks
to only the code required to make predictions, Neo reduces the memory footprint
for models by up to 10 times. Neo can optimize models for a variety of platforms,
which makes tuning a model for multiple deployments simple.

## Running the notebook
The use case for this post is an image classification task using a pre-trained ResNet
model that is fine-tuned for the food images within the hot dog / not hot dog
dataset. The notebook shows how to use Amazon SageMaker to fine-tune a pre-
trained convolutional neural network model, optimize the model using Neo, and
deploy the model and evaluate its latency in a variety of methods using Neo and EIA.