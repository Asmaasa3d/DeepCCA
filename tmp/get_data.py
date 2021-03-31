import numpy as np
from sklearn.model_selection import train_test_split


#data from DCCA model
from dcca import projected_feats,data_view_1 as data1,data_view_2 as data2

def main():
  d=generate_data()
  data =load_data(d)
  return data
def mappingLocIdxToPhysicalLoc(idx):
    return {
        1:  {'x':5, 'y':1 },
        2:  {'x':5, 'y':2 },
        3:  {'x':6, 'y':2 },
        4:  {'x':7, 'y':1 },
        5:  {'x':5, 'y':3 },
        6:  {'x':5, 'y':4 },
        7:  {'x':6, 'y':4 },
        8:  {'x':7, 'y':4 },
        9:  {'x':8, 'y':4 },
        10: {'x':5, 'y':5 },
        11: {'x':6, 'y':6 },
        12: {'x':6, 'y':7 },
        13: {'x':6, 'y':8 },
        14: {'x':7, 'y':6 },
        15: {'x':8, 'y':6 },
        16: {'x':9, 'y':6 },
        17: {'x':9, 'y':5 },
        18: {'x':9, 'y':4 },
        19: {'x':9, 'y':7 },
        20: {'x':8, 'y':7 },
        21: {'x':8, 'y':8 },
        22: {'x':9, 'y':8 },
        23: {'x':9, 'y':9 },
        24: {'x':9, 'y':10 },
        25: {'x':5, 'y':6 },
        26: {'x':5, 'y':7 },
        27: {'x':5, 'y':8 },
        28: {'x':5, 'y':9 },
        29: {'x':5, 'y':10 },
        30: {'x':5, 'y':11 },
        31: {'x':6, 'y':11 },
        32: {'x':7, 'y':11 },
        33: {'x':6, 'y':10 },
        34: {'x':7, 'y':10 },
        35: {'x':7, 'y':9 },
        36: {'x':6, 'y':9 },
        37: {'x':4, 'y':4 },
        38: {'x':3, 'y':4 },
        39: {'x':2, 'y':4 },
        40: {'x':1, 'y':4 },
        41: {'x':1, 'y':5 },
        42: {'x':1, 'y':6 },
        43: {'x':1, 'y':7 },
        44: {'x':1, 'y':8 },
        45: {'x':1, 'y':9 },
        46: {'x':1, 'y':10 },
        47: {'x':1, 'y':11 },
        48: {'x':2, 'y':11 },
        49: {'x':3, 'y':11 },
        50: {'x':3, 'y':10 },
        51: {'x':3, 'y':9 },
        52: {'x':3, 'y':8 },
        53: {'x':3, 'y':7 },
        54: {'x':3, 'y':6 },
        55: {'x':3, 'y':5 }
    }[idx]


 
def Euclidean_Dis(ID1,ID2):
  a=mappingLocIdxToPhysicalLoc(ID1)
  b=mappingLocIdxToPhysicalLoc(ID2)
  p1=[a['x'],a['y']]
  p2=[b['x'],b['y']]
  p1=np.asanyarray(p1)
  p2=np.asanyarray(p2)
  #https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
  return np.linalg.norm(p1-p2)



def generate_data():

  tmp=[]
  for i in range(1):
    
    for RP1 in range(1,56):
      for RP2 in range(1,56):
        #x=concatenate Pred_out[provider1]&Pred_out[provider2]
        cnt1=5
        for idx1,j in enumerate(data1[i][1]) :
          if j==RP1 and cnt1>0 :
            cnt1-=1
            cnt2=2
            for idx2,k in enumerate(data2[i][1]):
              if k==RP2 and cnt2>0:
                cnt2-=1
                x=projected_feats[i][0][idx1].tolist()+projected_feats[i][1][idx2].tolist()
                #y=distance between(point1,point2)
                y=Euclidean_Dis(j,k)
                #new_data append [x,y]
                tmp.append([x,y])
    return tmp
  
def load_data(data):
  x=[data[i][0] for i in range(len(data))]
  y=[float(np.array(data[i][1])) for i in range(len(data))]
  X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
  test_set_x, valid_set_x, test_set_y, valid_set_y = train_test_split(X_test,y_test,test_size=0.5)

  train_set_x = np.asarray(X_train, dtype='float32')
  train_set_y= np.asarray(y_train, dtype='float32') 
  test_set_x = np.asarray(test_set_x, dtype='float32')
  test_set_y = np.asarray(test_set_y, dtype='float32') 
  valid_set_x=np.asarray(valid_set_x, dtype='float32')
  valid_set_y = np.asarray(valid_set_y, dtype='float32')
  return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]         
