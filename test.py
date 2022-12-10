from __future__ import division
from __future__ import print_function

import time
import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('/content/drive/MyDrive/GAE/gae/train.py'), '..')))


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from pydist2.distance import pdist2
import numpy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from numpy import linalg
import glob


from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
emblist=[]

k=0
counter=0
# ext = ('.jpg')
# dirname ='/content/drive/MyDrive/datasetyale'
# for path, dirc, files in os.walk(dirname):
#   for name in files:
#       if name.endswith(ext):
#         print("hi")

temp=[]


files = glob.glob('./orl_dataset/person1/train_images/*.jpg', recursive = True)
for file in files:
  theFinalEmb=[]
  img_name=str(file.split('/')[3])
  print(img_name)
  adj, features, adjk = load_data(file)
  re_const=[]
  emb_holder=[]

  def get_roc_score(adjk,count,emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
    # print(emb)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = sigmoid(np.dot(emb, emb.T))
    re_const.append(emb)
    #print(emb)
    if(count%2==1):
      point1=np.array(re_const[0]) 
      point2=np.array(re_const[1])
      dist = np.linalg.norm(point1 - point2)
      #print("this is dist",dist)
      re_const.clear()
    if(count==999):
      emblist.append(emb)
      theFinalEmb.append(emb)
      # print("This is that of thsi",len(emblist))

  # print(adj_rec)
    #print(emb)
    # print("roc---{}".format(roc_auc_score(adjk,adj_rec)))
    # print("AP---{}".format(average_precision_score(adjk, adj_rec)))
    #print("This is the reconstructed data ",re_const)
  # Load data


  # Store original adjacency matrix (without diagonal entries) for later
  adj_orig = adj
  adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
  adj_orig.eliminate_zeros()
  #print((adjk),"This is ")
  # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
  # adj = adj_train

  if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

  # Some preprocessing
  adj_norm = preprocess_graph(adj)

  # Define placeholders
  placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
  }

  num_nodes = adj.shape[0]

  features = sparse_to_tuple(features.tocoo())
  num_features = features[2][1]
  features_nonzero = features[1].shape[0]

  # Create model
  model = 'gcn_vae'
  if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
  elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

  pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
  norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

  # Optimizer
  with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            model=model, num_nodes=num_nodes,
                            pos_weight=pos_weight,
                            norm=norm)

  # Initialize session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # cost_val = []
  # acc_val = []

  adj_label = adj + sp.eye(adj.shape[0])
  adj_label = sparse_to_tuple(adj_label)

  # def get_roc_score(emb=None):
  #   if emb is None:
  #       feed_dict.update({placeholders['dropout']: 0})
  #       emb = sess.run(model.z_mean, feed_dict=feed_dict)

  #   def sigmoid(x):
  #       return 1 / (1 + np.exp(-x))

  #   # Predict on test set of edges
  #   adj_rec = np.dot(emb, emb.T)
  #   re_const.append(adj_rec)
    
    # preds = []
    # pos = []
    # for e in edges_pos:
    #     preds.append(sigmoid(adj_rec[e[0], e[1]]))
    #     pos.append(adj_orig[e[0], e[1]])

    # preds_neg = []
    # neg = []
    # for e in edges_neg:
    #     preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
    #     neg.append(adj_orig[e[0], e[1]])

    # preds_all = np.hstack([preds, preds_neg])
    # labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    # roc_score = roc_auc_score(labels_all, preds_all)
    # ap_score = average_precision_score(labels_all, preds_all)
    # print("This is emb",emb)
    # return roc_score, ap_score


  cost_val = []
  acc_val = []
  val_roc_score = []

  adj_label = adj+ sp.eye(adj.shape[0])
  adj_label = sparse_to_tuple(adj_label)

  # Train model
  #i=0
  for epoch in range(FLAGS.epochs):
    #print(epoch,"this is epoch")
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
    adjk=np.array(adjk)
    # adjk=np.ceil(adjk)
    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    # roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    # val_roc_score.append(roc_curr)
    
    get_roc_score(adjk,(epoch))
    if(epoch==999):
      print("inside epoch",theFinalEmb)
    #print("Loss :{}\n Accuracy: {}".format(avg_cost,avg_accuracy))
    #print(avg_cost)
    #print("Epoch:", '%04d' % (epoch + 1),"train_loss=","{:.5f}".format(avg_cost),"recontstuct",re_const)
    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
    #       "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
    #       "val_ap=", "{:.5f}".format(ap_curr),
    #       "time=", "{:.5f}".format(time.time() - t))
  #print(epoch)
  #emb_holder.append(emblist)

  #print(temp2,temp1)
  filename='data/temp/output/test/'+img_name.split('.')[0]+'.csv'
  print(theFinalEmb[0])
  np.savetxt(filename, theFinalEmb[0], delimiter=',')
  counter=counter+1
  print("\n\nOptimization Finished!",filename,counter)
  #print("This is",len(emblist))
  count=0
  
  # for idx, x in enumerate(emblist):
  #     # temp=idx/5
  #     # print(idx,x)
  #     # print("Thhis",temp,count)
  #     # if(idx%5==0):
  #     #   count+=1
  #     # else:
  #       filename='sonikanewsubject'+str(int(idx/5))+"_"+str(idx%5)+'.csv'
  #       np.savetxt(filename, x, delimiter=',')
  #       count=count+1

  #point1=np.array(emblist[0]) 
  # point2=np.array(emblist[2])
  # dist = np.linalg.norm(point1 - point2)
  # print("this is sim dist",dist)
  # point1=np.array(emblist[0]) 
  # point2=np.array(emblist[1])
  # dist = np.linalg.norm(point1 - point2)
  # print("this is diff dist",dist)
  # point1=np.array(emblist[3]) 
  # point2=np.array(emblist[4])
  # dist = np.linalg.norm(point1 - point2)
  # print("this is same image differnet posediff dist",dist)
  #for i in range(0,)

    # roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
    # print('Test ROC score: ' + str(roc_score))
    # print('Test AP score: ' + str(ap_score))