import os
import os.path as path
import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf
from keras import backend as K

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from scipy import stats
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

def export_model(MODEL_NAME,input_node_names, output_node_name):
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
        'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
        "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

def model_input(input_node_name, keep_prob_node_name,input_height,input_width,num_channels,num_labels):
    x = tf.placeholder(tf.float32, shape=[None,input_width,num_channels, input_height], name=input_node_name)
    keep_prob = tf.placeholder_with_default(0.0,(), name=keep_prob_node_name)
    y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
    return x, keep_prob, y_

def build_LSTMCNN_model(x, keep_prob, y_, output_node_name,num_channels,num_labels,window_size,learning_rate,kernel_size):
    conv1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=kernel_size,strides=(1, 1),padding='valid', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=kernel_size, strides=(1, 1),padding='valid', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=kernel_size, strides=(1, 1),padding='valid', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=kernel_size, strides=(1, 1),padding='valid', activation=tf.nn.relu)

    x4=tf.reshape(conv4,(-1,34,num_channels*64))
    x5=tf.keras.layers.LSTM(128,return_sequences=True)(x4)
    x5 = tf.layers.dropout(x5, rate=keep_prob) 
    x6=tf.keras.layers.LSTM(128,return_sequences=False)(x5)
    x6=tf.reshape(x6,(-1,128))
    dropout2 = tf.layers.dropout(x6, rate=keep_prob)

    logits = tf.layers.dense(dropout2, num_labels)
    outputs = tf.nn.softmax(logits, name=output_node_name)
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # total_parameters = 0
    # for variable in tf.trainable_variables():
    #     # shape is an array of tf.Dimension
    #     shape = variable.get_shape()
    #     print(shape)
    #     variable_parameters = 1
    #     for dim in shape:
    #         variable_parameters *= dim.value
    #     print(variable_parameters)
    #     total_parameters += variable_parameters
    # print(total_parameters)
    # exit()

    return train_step, loss, accuracy,outputs

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def train(MODEL_NAME,EPOCHS,BATCH_SIZE,train_x,train_y,valid_x,valid_y,test_x,test_y,x, keep_prob, y_, train_step, loss, accuracy,saver,outputs):
    print("training start...")

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        tf.train.write_graph(K.get_session().graph_def, 'out',
            MODEL_NAME + '.pbtxt', True)

        for epoch in range(EPOCHS):
            for batch_x,batch_y in iterate_minibatches(train_x, train_y, BATCH_SIZE):  
                sess.run(train_step,feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
                    
            # out = sess.run(outputs, feed_dict={x:train_x, y_: train_y,keep_prob: 0.0})      
            # print("Epoch: ",epoch," Training Accuracy: ", sk.metrics.f1_score(np.argmax(out,1), np.argmax(train_y,1),average='weighted'))
            out = sess.run(outputs, feed_dict={x:valid_x, y_: valid_y,keep_prob: 0.0})        
            print("Valid Accuracy: ", sk.metrics.f1_score(np.argmax(out,1), np.argmax(valid_y,1),average='weighted'))

        out = sess.run(outputs, feed_dict={x:test_x, y_: test_y,keep_prob: 0.0})
        print ("Testing Accuracy:", sk.metrics.f1_score(np.argmax(out,1), np.argmax(test_y,1),average='weighted'))
        print("Confustion Matrix:")
        print(confusion_matrix(np.argmax(out,1),  np.argmax(test_y,1)))
        
        saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')    

    print("training finished!")


def main():
    if not path.exists('out'):
        os.mkdir('out')

    MODEL_NAME = 'mnist_convnet_gesture'

    input_height = 1
    input_width = 50
    num_labels = 24
    num_channels = 6

    batch_size = 100
    kernel_size = [5,1]

    learning_rate = 10e-4
    training_epochs = 15

    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'
    
    # train_x=np.load('../Datasets/OutSet/x_train.npy')
    # train_y=np.load('../Datasets/OutSet/y_train.npy')
    # valid_x=np.load('../Datasets/OutSet/x_valid.npy')
    # valid_y=np.load('../Datasets/OutSet/y_valid.npy')
    # test_x=np.load('../Datasets/OutSet/x_test.npy')
    # test_y=np.load('../Datasets/OutSet/y_test.npy')

    train_x=np.load('gdrive/My Drive/Datasets/OutSet/x_train.npy')
    train_y=np.load('gdrive/My Drive/Datasets/OutSet/y_train.npy')
    valid_x=np.load('gdrive/My Drive/Datasets/OutSet/x_valid.npy')
    valid_y=np.load('gdrive/My Drive/Datasets/OutSet/y_valid.npy')
    test_x=np.load('gdrive/My Drive/Datasets/OutSet/x_test.npy')
    test_y=np.load('gdrive/My Drive/Datasets/OutSet/y_test.npy')

    x, keep_prob, y_ = model_input(input_node_name=input_node_name, keep_prob_node_name=keep_prob_node_name,
        input_height=input_height,input_width=input_width,num_channels=num_channels,num_labels=num_labels)

    train_step, loss, accuracy,outputs = build_LSTMCNN_model(x=x, keep_prob=keep_prob,
        y_=y_, output_node_name=output_node_name,num_channels=num_channels,num_labels=num_labels,window_size=input_width,learning_rate=learning_rate
        ,kernel_size=kernel_size)
    saver = tf.train.Saver()

    train(MODEL_NAME,training_epochs,batch_size,train_x,train_y,valid_x,valid_y,test_x,test_y,x, keep_prob, y_, train_step, loss, accuracy,
        saver,outputs)

    export_model(MODEL_NAME,[input_node_name], output_node_name)

if __name__ == '__main__':
    main()