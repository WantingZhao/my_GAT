import time
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import argparse
import os
from models import GAT
from models import SpGAT
from utils import process
import json

dataset = 'dblp'
checkpt_file = 'pre_trained/cora/model_'+dataset+'cora.ckpt'

# training params
batch_size = 1
nb_epochs = 100
patience = 50
lr = 0.002  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
# model = GAT
model = SpGAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

sparse = True

for per_class in [200]:
    result_path = 'result/' + dataset + str(per_class) + '/'
    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    metapaths, metapaths_name, adjs, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(
        dataset,per_class)
    features, spars = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    features = features[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    all_metapaths_best = (0, 0, 0, 0)
    for metapath_id in range(len(metapaths)):
        best = (0, 0, 0, '')
        # 只测试APCPA
        if metapaths_name[metapath_id]!= ['AP','PC','CP','PA']:
            continue
        # if not(metapaths_name[metapath_id]!=[] and metapaths_name[metapath_id][0][0]==metapaths_name[metapath_id][-1][-1] and metapaths_name[metapath_id][0][0]=='P'):
        #     continue


        print("metapath=", metapaths_name[metapath_id], "-----------------------------")
        adj = adjs[metapath_id]
        if sparse:
            biases = process.preprocess_adj_bias(adj)
        else:
            adj = adj.todense()
            adj = adj[np.newaxis]
            biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

        with tf.Graph().as_default():
            with tf.name_scope('input'):
                ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
                if sparse:
                    # bias_idx = tf.placeholder(tf.int64)
                    # bias_val = tf.placeholder(tf.float32)
                    # bias_shape = tf.placeholder(tf.int64)
                    bias_in = tf.sparse_placeholder(dtype=tf.float32)
                else:
                    bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
                lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
                msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
                attn_drop = tf.placeholder(dtype=tf.float32, shape=())
                ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
                is_train = tf.placeholder(dtype=tf.bool, shape=())

            logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                     attn_drop, ffd_drop,
                                     bias_mat=bias_in,
                                     hid_units=hid_units, n_heads=n_heads,
                                     residual=residual, activation=nonlinearity)
            log_resh = tf.reshape(logits, [-1, nb_classes])
            lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
            msk_resh = tf.reshape(msk_in, [-1])
            loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
            accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

            train_op = model.training(loss, lr, l2_coef)

            saver = tf.train.Saver()

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            vlss_mn = np.inf
            vacc_mx = 0.0
            curr_step = 0

            with tf.Session() as sess:
                sess.run(init_op)

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

                for epoch in range(nb_epochs):
                    tr_step = 0
                    tr_size = features.shape[0]

                    while tr_step * batch_size < tr_size:
                        if sparse:
                            bbias = biases
                        else:
                            bbias = biases[tr_step * batch_size:(tr_step + 1) * batch_size]

                        _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                                                            feed_dict={
                                                                ftr_in: features[
                                                                        tr_step * batch_size:(
                                                                                                         tr_step + 1) * batch_size],
                                                                bias_in: bbias,
                                                                lbl_in: y_train[
                                                                        tr_step * batch_size:(
                                                                                                         tr_step + 1) * batch_size],
                                                                msk_in: train_mask[
                                                                        tr_step * batch_size:(
                                                                                                         tr_step + 1) * batch_size],
                                                                is_train: True,
                                                                attn_drop: 0.6, ffd_drop: 0.6})
                        train_loss_avg += loss_value_tr
                        train_acc_avg += acc_tr
                        tr_step += 1

                    vl_step = 0
                    vl_size = features.shape[0]

                    while vl_step * batch_size < vl_size:
                        if sparse:
                            bbias = biases
                        else:
                            bbias = biases[vl_step * batch_size:(vl_step + 1) * batch_size]
                        loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                         feed_dict={
                                                             ftr_in: features[
                                                                     vl_step * batch_size:(vl_step + 1) * batch_size],
                                                             bias_in: bbias,
                                                             lbl_in: y_val[
                                                                     vl_step * batch_size:(vl_step + 1) * batch_size],
                                                             msk_in: val_mask[
                                                                     vl_step * batch_size:(vl_step + 1) * batch_size],
                                                             is_train: False,
                                                             attn_drop: 0.0, ffd_drop: 0.0})
                        val_loss_avg += loss_value_vl
                        val_acc_avg += acc_vl
                        vl_step += 1

                    print('Epoch = %d, Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                          (epoch, train_loss_avg / tr_step, train_acc_avg / tr_step,
                           val_loss_avg / vl_step, val_acc_avg / vl_step))

                    # 保存最好的valid acc
                    if val_acc_avg > best[0]:
                        ts_size = features.shape[0]
                        ts_step = 0
                        ts_loss = 0.0
                        ts_acc = 0.0

                        show_feas=[]
                        test_y=[]

                        while ts_step * batch_size < ts_size:
                            if sparse:
                                bbias = biases
                            else:
                                bbias = biases[ts_step * batch_size:(ts_step + 1) * batch_size]
                            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                                                             feed_dict={
                                                                 ftr_in: features[
                                                                         ts_step * batch_size:(
                                                                                                          ts_step + 1) * batch_size],
                                                                 bias_in: bbias,
                                                                 lbl_in: y_test[
                                                                         ts_step * batch_size:(
                                                                                                          ts_step + 1) * batch_size],
                                                                 msk_in: test_mask[
                                                                         ts_step * batch_size:(
                                                                                                          ts_step + 1) * batch_size],
                                                                 is_train: False,
                                                                 attn_drop: 0.0, ffd_drop: 0.0})


                            tensor_dict = tf.get_default_graph().get_tensor_by_name('concat:0')
                            output = sess.run(tensor_dict, feed_dict={
                                                                 ftr_in: features[
                                                                         ts_step * batch_size:(
                                                                                                          ts_step + 1) * batch_size],
                                                                 bias_in: bbias,
                                                                 lbl_in: y_test[
                                                                         ts_step * batch_size:(
                                                                                                          ts_step + 1) * batch_size],
                                                                 msk_in: test_mask[
                                                                         ts_step * batch_size:(
                                                                                                          ts_step + 1) * batch_size],
                                                                 is_train: False,
                                                                 attn_drop: 0.0, ffd_drop: 0.0})

                            output=output[0]
                            print('batch_feature---------------', output.shape,test_mask[0].shape)
                            output = [z for z, mask in zip(output, test_mask[0]) if mask]
                            np.save("gat_show.npy", output)
                            output = [z for z, mask in zip(y_test[0], test_mask[0]) if mask]
                            np.save("gat_show_label.npy", np.array(output).argmax(axis=1))



                            ts_loss += loss_value_ts
                            ts_acc += acc_ts
                            ts_step += 1


                        print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)
                        best = (val_acc_avg, ts_acc / ts_step, epoch, str(metapaths_name[metapath_id]))
                        result = "temp---------at_best_validate: valid accuracy=%.5f, test accuracy=%.5f, epoch=%d, metapath=%s" % (
                            best[0], best[1], best[2], best[3])
                        print(result)
                        with open(result_path + "result_%s.txt" % str(metapaths_name[metapath_id]), 'w') as fout:
                            fout.write(json.dumps(result))

                    if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                        if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                            vacc_early_model = val_acc_avg / vl_step
                            vlss_early_model = val_loss_avg / vl_step
                            saver.save(sess, checkpt_file)
                        vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                        vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                        curr_step = 0
                    else:
                        curr_step += 1
                        if curr_step == patience:
                            print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                            print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ',
                                  vacc_early_model)
                            break

                    train_loss_avg = 0
                    train_acc_avg = 0
                    val_loss_avg = 0
                    val_acc_avg = 0

                saver.restore(sess, checkpt_file)

                ts_size = features.shape[0]
                ts_step = 0
                ts_loss = 0.0
                ts_acc = 0.0

                while ts_step * batch_size < ts_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[ts_step * batch_size:(ts_step + 1) * batch_size]
                    loss_value_ts, acc_ts = sess.run([loss, accuracy],
                                                     feed_dict={
                                                         ftr_in: features[
                                                                 ts_step * batch_size:(ts_step + 1) * batch_size],
                                                         bias_in: bbias,
                                                         lbl_in: y_test[
                                                                 ts_step * batch_size:(ts_step + 1) * batch_size],
                                                         msk_in: test_mask[
                                                                 ts_step * batch_size:(ts_step + 1) * batch_size],
                                                         is_train: False,
                                                         attn_drop: 0.0, ffd_drop: 0.0})
                    ts_loss += loss_value_ts
                    ts_acc += acc_ts
                    ts_step += 1

                print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

                sess.close()

                result = "at_best_validate: valid accuracy=%.5f, test accuracy=%.5f, epoch=%d, metapath=%s" % (
                    best[0], best[1], best[2], best[3])
                print(result)
                with open(result_path + "result_%s.txt" % str(metapaths_name[metapath_id]), 'w') as fout:
                    fout.write(json.dumps(result))

                if best[1] > all_metapaths_best[1]:
                    all_metapaths_best = best

        result = "best result: valid accuracy=%.5f, test accuracy=%.5f, epoch=%d, metapath=%s" % (
            all_metapaths_best[0], all_metapaths_best[1], all_metapaths_best[2], all_metapaths_best[3])
        print(result)
        with open(result_path + "best_result.txt", 'w') as fout:
            fout.write(json.dumps(result))
