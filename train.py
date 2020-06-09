import os
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from datetime import datetime
from mc_cnn_brunch import Net
from data_generator import ImagePatchesGenerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="training of MC-CNN")

parser.add_argument("-g", "--gpu", type=str, default="0,1,2,3,4,5,6", help="gpu id to use, \
                    multiple ids should be separated by commons(e.g. 0,1,2,3)")
parser.add_argument("-ps", "--patch_size", type=int, default=11, help="length for height/width of square patch")
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="mini-batch size")
parser.add_argument("-mr", "--margin", type=float, default=0.3, help="margin in hinge loss")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument("-bt", "--beta", type=int, default=0.9, help="momentum")

parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from. \
                   if None(default), model is initialized using default methods")

parser.add_argument("--start_epoch", type=int, default=3, help="start epoch for training(inclusive)")
parser.add_argument("--end_epoch", type=int, default=14, help="end epoch for training(exclusive)")
parser.add_argument("--print_freq", type=int, default=1, help="summary info(for tensorboard) writing frequency(of batches)")
parser.add_argument("--save_freq", type=int, default=1, help="checkpoint saving freqency(of epoches)")
parser.add_argument("--val_freq", type=int, default=1, help="model validation frequency(of epoches)")

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def main():
    args = parser.parse_args()

    # GPU preparation
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # directiory preparation
    tensor_graph_data_path = r'./logs_mr30'   # args.tensorboard_dir
    checkpoint_path = r'./check_points_mr30'   # args.checkpoint_dir
    test_mkdir(tensor_graph_data_path)
    test_mkdir(checkpoint_path)

    train_dataset  = ImagePatchesGenerator(r'./training_data_set/left/{}.pfm',
                                          r'./training_data_set/right_pos/{}.pfm',
                                          r'./training_data_set/right_neg/{}.pfm',
                                          0,
                                          1898240,
                                          patch_size=(11, 11),
                                          shuffle=True)
    val_dataset  = ImagePatchesGenerator(r'./training_data_set/left/{}.pfm',
                                          r'./training_data_set/right_pos/{}.pfm',
                                          r'./training_data_set/right_neg/{}.pfm',
                                          1898359,
                                          1898359+191232,
                                          patch_size=(11, 11),
                                          shuffle=False)

    patch_height = args.patch_size
    patch_width = args.patch_size
    batch_size = args.batch_size
    train_batches_per_epoch = train_dataset.dataset_size
    val_batches_per_epoch = val_dataset.dataset_size

    '''   begin to construct the model   '''
    # tensorflow placeholder for graph input
    leftx = tf.placeholder(shape=[batch_size, patch_height, patch_width, 1], dtype=tf.float32)
    rightx_pos = tf.placeholder(shape=[batch_size, patch_height, patch_width, 1], dtype=tf.float32)
    rightx_neg = tf.placeholder(shape=[batch_size, patch_height, patch_width, 1], dtype=tf.float32)

    left_brunch = Net(leftx, input_patch_size=patch_height, num_of_conv_layers=5, num_of_conv_feature_maps=64, batch_size=batch_size, is_branch=False)
    right_brunch_pos = Net(rightx_pos, input_patch_size=patch_height, num_of_conv_layers=5, num_of_conv_feature_maps=64, batch_size=batch_size, is_branch=True)
    right_brunch_neg = Net(rightx_neg, input_patch_size=patch_height, num_of_conv_layers=5, num_of_conv_feature_maps=64, batch_size=batch_size, is_branch=True)

    featuresl = tf.squeeze(left_brunch.features, [1, 2])
    featuresr_pos = tf.squeeze(right_brunch_pos.features, [1, 2])
    featuresr_neg = tf.squeeze(right_brunch_neg.features, [1, 2])

    with tf.name_scope('correlation'):
        cosine_pos = tf.reduce_sum(tf.multiply(featuresl, featuresr_pos), axis=-1)
        cosine_neg = tf.reduce_sum(tf.multiply(featuresl, featuresr_neg), axis=-1)

    with tf.name_scope('hinge_loss'):
        margin = tf.ones(shape=[batch_size], dtype=tf.float32)*args.margin
        loss = tf.reduce_mean(tf.maximum(0.0, margin - cosine_pos + cosine_neg))

    # with tf.name_scope('train'):
    var_list = tf.trainable_variables()
    for var in var_list:
        print('{}'.format(var.name))
    # gradients = list(zip(tf.gradients(loss, var_list), var_list))
    factor = tf.placeholder(tf.float32, [])
    optimizer = tf.train.MomentumOptimizer(args.learning_rate/factor, args.beta)
    gradients = optimizer.compute_gradients(loss, var_list)
    train = optimizer.apply_gradients(grads_and_vars=gradients)

    with tf.name_scope('training_metric'):
        training_summary = []
        training_summary.append(tf.summary.scalar('hinge_loss', loss))
        training_merged_summary = tf.summary.merge(training_summary)

    with tf.name_scope('val_metric'):
        val_summary = []
        val_loss = tf.placeholder(tf.float32, [])
        val_summary.append(tf.summary.scalar('val_hinge_loss', val_loss))
        val_merged_summary = tf.summary.merge(val_summary)

    # tensor graph, data writer
    writer = tf.summary.FileWriter(tensor_graph_data_path)

    # to save the model's data
    saver = tf.train.Saver(max_to_keep=10)

    ###########################################################################################
    # # # train # # #
    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        sess.run(tf.initialize_all_variables())

        if args.resume is None:
            writer.add_graph(sess.graph)
        else:
            saver.restore(sess, args.resume)

        print('traing_batches_per_epoch:{}, val_batches_per_epoch:{}'.format(train_batches_per_epoch, \
                                                                             val_batches_per_epoch))
        print('{} start training...'.format(datetime.now()))
        print('{} open Tensorboard at --logdir {}'.format(datetime.now(), tensor_graph_data_path))

        for epoch in range(args.start_epoch, args.end_epoch):
            print('{} Epoch number: {}'.format(datetime.now(), epoch+1))

            for batch in tqdm(range(train_batches_per_epoch)):
                batch_left, batch_right_pos, batch_right_neg = train_dataset.next_batch()
                fac = 1
                if epoch+1 > 10:
                    fac = 10
                sess.run(train, feed_dict={leftx: batch_left, rightx_pos: batch_right_pos, rightx_neg: batch_right_neg, factor: fac})

            if (epoch+1) % args.print_freq == 0:
                s = sess.run(training_merged_summary, \
                             feed_dict={leftx: batch_left, rightx_pos: batch_right_pos, rightx_neg: batch_right_neg, factor: fac})
                writer.add_summary(s, (epoch+1)*train_batches_per_epoch)

            if (epoch+1) % args.save_freq == 0:
                print("{} Saving checkpoint of model...".format(datetime.now()))
                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)
                # print(save_path)

            if (epoch+1) % args.val_freq == 0:
                print('{} Start valization'.format(datetime.now()))
                val_ls = 0
                for _ in tqdm(range(val_batches_per_epoch)):
                    batch_left, batch_right_pos, batch_right_neg = val_dataset.next_batch()
                    result = sess.run(loss, feed_dict= \
                        {leftx: batch_left, rightx_pos: batch_right_pos, rightx_neg: batch_right_neg, factor: fac})
                    val_ls += result
                val_ls = val_ls / (1. * val_batches_per_epoch)

                print('validation loss: {}'.format(val_ls))
                s = sess.run(val_merged_summary, feed_dict={val_loss: np.float32(val_ls)})
                writer.add_summary(s, train_batches_per_epoch*(epoch + 1))

            train_dataset.reset_pointer()
            val_dataset.reset_pointer()


if __name__ =="__main__":
    main()


