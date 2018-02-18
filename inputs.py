for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import deque

SKIP_IMAGES_FROM_DATA_SET = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN =500

def read_data(im_dir, la_dir, skip, select):
    print("loading data ... ")
    

    la_folders = ['vfist.csv', 'fist.csv', 'flat.csv'] 
    im_folders = ['vfist', 'fist', 'flat']

    la_im = pd.DataFrame()
    data = deque()
    labels = []
    for la,im in zip(la_folders,im_folders):
        la_im = pd.read_csv(os.path.join(la_dir,la),header=None, index_col=None)
        la_im = la_im.loc[skip:,:]
        labels.extend(list(map(np.float32,la_im.loc[:,0])))
        images = os.listdir(os.path.join(im_dir,im))
        for i, img in enumerate(la_im.loc[:,1]):
            if 'image_' + img + '.jpeg' in images:
                image = np.asarray(Image.open(os.path.join(im_dir,im, 'image_'+img+'.jpeg').convert('L')))
                data.append(image.ravel())
                #print(os.path.join(im_dir,im, 'image_'+img+'.jpeg'))
                print(len(data),len(data[0]))
                
                
    data = np.array(data).reshape(-1, 480, 390, 1)
    labels = np.array(labels)
    
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    idx = idx[:select]
    data, labels = data[idx], labels[idx]
    
    print('Reading data done!')
    
    #!!!!!!!!!!!!!!!!1note that select must be less than len(labels)!!!!!!!!!!!!!!!!!
    select = np.random.randint(0,len(labels)-1,select)
    data = data[select]
    labels = labels[select]
    
    return (data, labels)
   
                
def data_preprocessing(im_dir, la_dir, skip=SKIP_IMAGES_FROM_DATA_SET, select=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, phase='train'):
    print("loading data ... ")
   
    la_folders = ['vfist.csv', 'fist.csv', 'flat.csv'] 
    im_folders = ['vfist', 'fist', 'flat']

    la_im = pd.DataFrame()
    labels = []
    im_data = []
    i = 0
    for la,im in zip(la_folders,im_folders):
        la_im = pd.read_csv(os.path.join(la_dir,la),header=None, index_col=None)
        la_im = la_im.loc[skip:,:]
        images = os.listdir(os.path.join(im_dir,im))
        
        for lab,img in zip(la_im.loc[:,0],la_im.loc[:,1]):
            if 'image_' + img + '.jpeg' in images:
                im_data.append(np.array(Image.open(os.path.join(im_dir,im)+'/image_' + img + '.jpeg')).ravel())
                labels.append(lab)
                #print(i)
                #i += 1

    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    idx = idx[:select]
    labels, im_data = np.array(labels), np.array(im_data)
    im_data, labels = im_data[idx], labels[idx]
    im_data = im_data.reshape(-1, 480, 390, 3)


    print('****im_list and labels length are:****',im_data.shape,labels.shape)
    
    split = int(0.8 * len(im_data))
    if phase =='train':
        im_data_train, im_data_val = np.split(im_data, [split])
        labels_train, labels_val =  np.split(labels, [split])
        return (im_data_train, im_data_val , labels_train, labels_val)
    else:
        return (im_data, labels)        
        
        
    
    '''

    la_im_flat_dir = '/media/yazdan/061C4B551C4B3F45/Yazdan/Research/projct/Data/New_data/Im_la/ambient'
    im_dir = '/media/yazdan/061C4B551C4B3F45/Yazdan/Research/projct/Data/bkg_static/ambient/flat'
    images = os.listdir(im_dir)
    la_im = pd.read_csv(os.path.join(la_im_flat_dir,'flat.csv'),header=None, index_col=None)
    la_im = la_im.loc[2000:,:]
    #la_im.loc[:,0] = la_im.loc[:,0].map(lambda x:x.strip())
    data = []

    for i, im in enumerate(la_im.loc[:,1]):
        #print(im+'.jpeg')
        if 'image_'+im+'.jpeg' in images:
            img = np.asarray(Image.open(os.path.join(im_dir, 'image_'+im+'.jpeg')))
            data.append(img.ravel())

    labels = np.array(list(map(np.float32,la_im.loc[:,0])))
    #print(list(labels))
    #print(np.array(data).shape)    

    
    x_train, x_test = np.vsplit(data, [len(data)//2])
    y_train, y_test =  np.split(labels,[len(data)//2])
     

    
    x_train = x_train.reshape(-1, 480, 390, 3).astype(np.float32)
    x_test = x_test.reshape(-1, 480, 390, 3).astype(np.float32)

  
    #train_set = batch_iterator(it.cycle(zip(x_train, y_train)), batch_size, cycle=True, batch_fn=lambda x: zip(*x))
    #test_set = (x_test, y_test)
    train_set = (x_train, y_train)
    test_set = (x_test, y_test)

    
    return train_set, test_set

'''

'''if __name__ == '__main__':
    
    trn_im_dir = 'bkg_static/ambient'
    trn_la_dir = 'Im_la/ambient'
    tst_im_dir = 'add_blue_light'
    tst_la_dir = 'add_blue_light'
    train = read_data(trn_im_dir, trn_la_dir, skip=100)
    test   = read_data(tst_im_dir, tst_la_dir,skip=2000)'''
