from __future__ import print_function

import json
import logging
import os
import shutil
import time
import warnings
import argparse
from glob import glob
import re

import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import io

##############################################
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data.vision import transforms
##############################################



# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--resnet_size', type=str, default='101')
    parser.add_argument('--log_interval', type=int, default=1)

    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--channel_input_dirs', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    return parser.parse_args()

    
def train(current_host, hosts, num_gpus, log_interval, channel_input_dirs, 
          batch_size, epochs, learning_rate, momentum, wd, resnet_size):
    
    print("Using Resnet {} model".format(resnet_size))
    model_options = {'18':models.resnet18_v2, '34':models.resnet34_v2, 
                     '50':models.resnet50_v2, '101':models.resnet101_v2, '152':models.resnet152_v2}
    
    if resnet_size not in model_options:
        raise Exception('Resnet size must be one of 18, 34, 50, 101, or 152')
        
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync'
    
    if num_gpus > 0:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
        
    print(ctx)
    selected_model = model_options[resnet_size]
    pretrained_net = selected_model(ctx=ctx, pretrained=True)
    net = selected_model(ctx=ctx, pretrained=False, classes=2)  # Changed classes to 2
    net.features = pretrained_net.features

    part_index = 0
    for i, host in enumerate(hosts):
        if host == current_host:
            part_index = i
            break

  
    data_dir = channel_input_dirs
    os.mkdir('/opt/ml/checkpoints')
    CHECKPOINTS_DIR = '/opt/ml/checkpoints'
    checkpoints_enabled = os.path.exists(CHECKPOINTS_DIR)
    
    ############################################### 
    #train = ImageFolderDataset('./hotdog_not_hotdog/train')
    #test = ImageFolderDataset('./hotdog_not_hotdog/test')
    
    print("The data dir is: ")
    print(data_dir)
    print(os.listdir(data_dir))
    print(os.listdir('/opt/ml/input/data/training/train'))
        
    
    train = ImageFolderDataset('/opt/ml/input/data/training/train')
    test = ImageFolderDataset('/opt/ml/input/data/training/test')
   
    
    print("Loaded Image Folders")
    
    transform_func = transforms.Compose([
                                   transforms.Resize(size=(256)),
                                   transforms.CenterCrop(size=(224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.49139969, 0.48215842, 0.44653093],
                                                         std=[0.20220212, 0.19931542, 0.20086347])])
    
    train_transformed = train.transform_first(transform_func)
    test_transformed = test.transform_first(transform_func)
    
    print("Transformed Training and Test Files")
    
    train_data = gluon.data.DataLoader(train_transformed, batch_size=32, shuffle=True, num_workers=1)
    test_data = gluon.data.DataLoader(test_transformed, batch_size=32, num_workers=1)
    
    print("Initialized Batching Operation")

    net.initialize(mx.init.Xavier(), ctx=ctx)
    print("Initialized Model")

    # Trainer is for updating parameters with gradient.
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    #trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            optimizer_params={'learning_rate': learning_rate, 'momentum': momentum, 'wd': wd},
                            kvstore=kvstore)
    metric = mx.metric.Accuracy()
    net.hybridize()

    best_loss = 5.0
    for epoch in range(epochs):
        # training loop (with autograd and trainer steps, etc.)
        cumulative_train_loss = mx.nd.zeros(1, ctx=ctx)
        training_samples = 0
        metric.reset()
        for batch_idx, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            outputs = []
            with ag.record():
                output = net(data)

                loss = criterion(output, label)
                outputs.append(output)
            loss.backward()
            trainer.step(data.shape[0])
            metric.update(label, output)
            cumulative_train_loss += loss.sum()
            training_samples += data.shape[0]
        train_loss = cumulative_train_loss.asscalar()/training_samples
        name, train_acc = metric.get()
        print("done training section")
        
        # validation loop
        cumulative_valid_loss = mx.nd.zeros(1, ctx)
        valid_samples = 0
        metric.reset()
        for batch_idx, (data, label) in enumerate(test_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = net(data)
            loss = criterion(output, label)
            cumulative_valid_loss += loss.sum()
            valid_samples += data.shape[0]
            metric.update(label, output)
        valid_loss = cumulative_valid_loss.asscalar()/valid_samples
        name, val_acc = metric.get()

        print("Epoch {}, training loss: {:.2f}, validation loss: {:.2f}, train accuracy: {:.2f}, validation accuracy: {:.2f}".format(epoch, train_loss, valid_loss, train_acc, val_acc))
        
        # only save params on primary host
        if checkpoints_enabled and current_host == hosts[0]:
            if valid_loss < best_loss:
                best_loss = valid_loss
                logging.info('Saving the model, params and optimizer state')
                net.export(CHECKPOINTS_DIR + "/%.4f-model"%(best_loss), epoch)
                save(net, CHECKPOINTS_DIR)
                trainer.save_states(CHECKPOINTS_DIR + '/%.4f-hotdog-%d.states'%(best_loss, epoch))
                 
    return net


def save(net, model_dir):
    # model_dir will be empty except on primary container
    files = os.listdir(model_dir)
    print(files)
    if files:
        files = sorted(os.listdir(model_dir))
        best_model_params = []
        for f in files:
            if f.endswith('.params'):
                best_model_params.append(f)
        best_model = best_model_params[0]
        
        best_model_symbol = []
        for f in files:
            if f.endswith('.json'):
                best_model_symbol.append(f)
        best_symbol = best_model_symbol[0]
        
        os.rename(os.path.join(model_dir, best_model), os.path.join(model_dir, best_model.split('-',1)[1]))
        os.rename(os.path.join(model_dir, best_symbol), os.path.join(model_dir, best_symbol.split('-',1)[1]))
        #clean up old file
        older_model = glob('/opt/ml/model/*.params')
        for f in older_model:
            os.remove(f)
        shutil.copyfile(os.path.join('/opt/ml/checkpoints/', best_model.split('-',1)[1]), os.path.join('/opt/ml/model/', best_model.split('-',1)[1]))
        shutil.copyfile(os.path.join('/opt/ml/checkpoints/', best_symbol.split('-',1)[1]), os.path.join('/opt/ml/model/', best_symbol.split('-',1)[1]))
        print(os.listdir('/opt/ml/model/'))



def get_data(path, augment, num_cpus, batch_size, data_shape, resize=-1, num_parts=1, part_index=0):
    return mx.image.ImageIter(
        path_root=path,
        resize=resize,
        data_shape=data_shape,
        batch_size=batch_size,
        rand_crop=augment,
        rand_mirror=augment,
        preprocess_threads=num_cpus,
        num_parts=num_parts,
        part_index=part_index)


def load_model(prefix, context):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs = int(re.search(string=glob(f'{prefix}-[0-9]*')[0], pattern=r'(\d+)').group(0))
        resnet_model = mx.mod.Module.load(prefix,epoch=epochs,context=context)
        resnet_model.bind(for_training=False, data_shapes=[('data', (1,3,512,512))])
        return resnet_model

    
def get_test_data(num_cpus, test_dir, batch_size, data_shape, resize=-1):
    return get_data(test_dir, False, num_cpus, batch_size, data_shape, resize, 1, 0)


def get_train_data(num_cpus, train_dir, batch_size, data_shape, resize=-1, num_parts=1, part_index=0):
    return get_data(train_dir, True, num_cpus, batch_size, data_shape, resize, num_parts,
                    part_index)


def test(ctx, net, test_data):
    test_data.reset()
    metric = mx.metric.Accuracy()

    for i, batch in enumerate(test_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    
    if os.environ.get('SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT') == 'true':
        ctx = mx.eia()
        print("Placing Model on {} context".format(ctx))
        prefix = f"{model_dir}/model"
        net = load_model(prefix, ctx)
    elif mx.context.num_gpus() > 0:  
        ctx = mx.gpu()
        print("Placing Model on {} context".format(ctx))
        prefix = f"{model_dir}/model"
        net = load_model(prefix, ctx)
    else:
        ctx = mx.cpu()
        print("Placing Model on {} context".format(ctx))
        prefix = f"{model_dir}/model"
        net = load_model(prefix, ctx)
    return net


def transform_fn(net, data, input_content_type='application/x-npy', output_content_type='application/json'):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we use numpy for input and json for output
    import numpy as np
    import io
    import base64
        
    with warnings.catch_warnings():
        img = json.loads(data)
        
    if os.environ.get('SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT') == 'true':
        ctx = mx.eia()
        ndarray = mx.nd.array(img, ctx)
    elif mx.context.num_gpus() > 0:  
        ctx = mx.gpu()
        ndarray = mx.nd.array(img, ctx)
    else:
        ctx = mx.cpu()
        ndarray = mx.nd.array(img, ctx)
            
    class_dict = {0:"not_hot_dog", 1:"hot_dog"}
    output = net.predict(ndarray).asnumpy()
    result = np.squeeze(output)
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)
    result_class = np.argmax(result)
    response_body = json.dumps({"predicted_class":class_dict[result_class], "confidence":str(result[result_class])})
    return response_body, output_content_type
    
    
### NOTE: this function cannot use MXNet
def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import io

    logging.info('Invoking user-defined pre-processing function')

    if content_type != 'application/vnd+python.numpy+binary':
        raise RuntimeError('Content type must be application/vnd+python.numpy+binary')

    f = io.BytesIO(payload)
    return np.load(f)


### NOTE: this function cannot use MXNet
def neo_postprocess(result):
    import logging
    import numpy as np
    import json

    logging.info('Invoking user-defined post-processing function')

    # Softmax (assumes batch size 1)
    class_dict = {0:"not_hot_dog", 1:"hot_dog"}
    result = np.squeeze(result)
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)
    result_class = np.argmax(result)
    response_body = json.dumps({"predicted_class":class_dict[result_class], "confidence":str(result[result_class])})
    content_type = 'application/json'

    return response_body, content_type

if __name__ == '__main__':
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    args = parse_args()
    train(args.current_host, args.hosts, num_gpus, args.log_interval, args.channel_input_dirs, args.batch_size, 
          args.epochs, args.learning_rate, args.momentum, args.wd, args.resnet_size)