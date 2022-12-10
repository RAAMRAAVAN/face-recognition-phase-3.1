import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from networkx import gnm_random_graph
import numpy
import json
import numpy as np
import itertools
import glob
import os
import mediapipe as mp

from scipy.sparse import  lil_matrix
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.stats import pearsonr
import copy as copy

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(temp1):
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_face_mesh = mp.solutions.face_mesh
  list1=[]
  newlist=[]
  list2=[]
  dist=[]
  dist1=[]
  # For static images:
  #path=glob.glob("/content/drive/MyDrive/testimage/*.jpg")
  # path='data/neutral_faces/'+str(temp1)
  path=temp1

  print(path)
  with mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5) as face_mesh:
      image = cv2.imread(path)
      
    # print("filename",path)
      # Convert the BGR image to RGB before processing.
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      # Print and draw face mesh landmarks on the image.
      if not results.multi_face_landmarks:
        pass
      annotated_image = image.copy()
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
      cv2.imwrite('/tmp/annotated_image' + '.jpg', annotated_image) 
      i=0
      for face in results.multi_face_landmarks[0].landmark:
        list1.append(results.multi_face_landmarks[0].landmark[i].x)
        list1.append(results.multi_face_landmarks[0].landmark[i].y)
        i=i+1
      with open('/content/drive/MyDrive/gaewithedit/gae-master/gae-master/gae/listfile.txt', 'r') as filehandle:
          points = json.load(filehandle)
      n=len(points)
      for i in range(0,n):
        k=points[i][0]
        #print(k)
        l=points[i][1]
        k=k*2
        l=l*2
        x1=list1[k]
        y1=list1[k+1]
        x2=list1[l]
        y2=list1[l+1]
        shape = image.shape 
        relative_x = int(x1 * shape[1])
        relative_y = int(y1 * shape[0])
        relative_z = int(x2 * shape[1])   
        relative_c = int(y2* shape[0])
        cv2.circle(image, (relative_x, relative_y), radius=1, color=(0, 0, 255), thickness=8) 
        cv2.line(image,(relative_x,relative_y),(relative_z,relative_c),(0,255,0),1)
      k=0
      for i in range (0,n-2):
        temp=[]
        l=points[i][1]
        l=l*2
        k=points[i][0]
        k=k*2
        temp.append(points[i][0])
        temp.append(points[i][1])
        num=((((list1[k]-list1[l])**2 + (list1[k+1]-list1[l+1])**2 ) ** 0.5))
        num1=num
        temp.append(num1)
        dist.append(temp)
      n=len(dist)
      G=nx.Graph()
      graph1 = os.path.split(temp1)
      graph2=graph1[1].split('.')[0]
      for i in range(n):
        arg1=dist[i][0]
        arg2=dist[i][1]
        arg3=dist[i][2]
        G.add_edge(arg1,arg2,weight=arg3)
      nx.write_weighted_edgelist(G,'allyalefacedataset_graph/'+graph2+'.csv')
      adjm = nx.adjacency_matrix(G, nodelist=None, dtype=None, weight='weight')

      adjk = csr_matrix.todense(adjm)
      adj = csr_matrix(adjm)
      #print(adj)
      #print(dist)
      features=[]
      features = sp.identity(adj.shape[0])
      features = lil_matrix(features)

      cv2_imshow(image) 
      cv2.waitKey(0)
      cv2.destroyAllWindows()

      return adj, features, adjk
