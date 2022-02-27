from pandas import read_csv, read_excel
from numpy import array
from matplotlib import pyplot as plt
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, LSTM, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics  import MeanSquaredError
import tensorflow as tf
from os.path import exists
from os import listdir, rename, environ
from pathlib import Path
from warnings import filterwarnings
filterwarnings('ignore')

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging_mode="info"

class Logger:
    def __init__(self,fname="log.txt"):
        self.fname=fname
    def log(self,data):
        if logging_mode=="debug":
            with open(self.fname,'a') as log_file:
                log_file.write(f"[ {datetime.now()} ] - ")
                log_file.write(data)
                log_file.write('\n')

log=Logger()

class Model:
    def __init__(self):
        self.model=Sequential()
    def CNN(self,input_shape=None,stride=(1,1),dilation=(1,1),kernel_n=3,pooling_size=(2,2),dropout_p=0.2,n_output=2,learning_rate=1e-3):
        log.log("CNN Model Initilaizing...")
        if input_shape!=None:
            self.kernel_n=kernel_n
            self.input_shape=input_shape 
            self.stride=stride      # skips, kernel makes at every convolution
            self.dilation=dilation  # kernel coverage
            self.pooling_size=pooling_size
            self.dropout_p=dropout_p
            self.n_output=n_output
            self.learning_rate=learning_rate
            self.model.add(Conv2D(filters=16,kernel_size=self.kernel_n,activation='relu',
            padding='same',input_shape=self.input_shape,strides=self.stride, dilation_rate=self.dilation))
            #self.model.add(MaxPool2D(pool_size=self.pooling_size))
            self.model.add(Conv2D(filters=32,kernel_size=self.kernel_n,activation='relu',
            padding='same',strides=self.stride, dilation_rate=self.dilation))
            #self.model.add(MaxPool2D(pool_size=self.pooling_size))
            #self.model.add(Conv2D(filters=64,kernel_size=self.kernel_n,activation='relu',
            #padding='same',strides=self.stride, dilation_rate=self.dilation))
            #self.model.add(MaxPool2D(pool_size=self.pooling_size))
            self.model.add(Conv2D(filters=128,kernel_size=self.kernel_n,activation='relu',
            padding='same',strides=self.stride, dilation_rate=self.dilation))
            #self.model.add(MaxPool2D(pool_size=self.pooling_size))
            self.model.add(Flatten())
            self.model.add(Dense(units=self.input_shape[0] * 128,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            #self.model.add(Dense(units=128,activation='relu'))
            #self.model.add(Dropout(self.dropout_p))
            #self.model.add(Dense(units=64,activation='relu'))
            #self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=32,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=16,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=self.n_output))
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),loss=MeanAbsoluteError(),metrics=[MeanSquaredError()])
        log.log("CNN Model Initialized!")
    def LSTM(self,input_shape=None,dropout_p=0.2,n_output=2,learning_rate=1e-3):
        log.log("LSTM Model Initilaizing...")
        if input_shape!=None:
            self.input_shape=input_shape
            self.dropout_p=dropout_p
            self.n_output=n_output
            self.learning_rate=learning_rate
            self.model.add(LSTM(8,dropout=self.dropout_p,input_shape=self.input_shape,return_sequences=True))
            self.model.add(LSTM(16,dropout=self.dropout_p,return_sequences=True))
            self.model.add(LSTM(32,dropout=self.dropout_p,return_sequences=True))
            self.model.add(LSTM(64,dropout=self.dropout_p,return_sequences=True))
            self.model.add(Flatten())
            self.model.add(Dense(units=64,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=32,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=16,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=8,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=self.n_output))
            #print(self.model.summary())
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),loss=MeanAbsoluteError(),metrics=[MeanSquaredError()])
        log.log("LSTM Model Initialized!")
    def LSTM_CNN_HYBRID(self,input_shape=None,stride=(1,1),dilation=(1,1),kernel_n=3,pooling_size=(2,2),dropout_p=0.2,n_output=2,learning_rate=1e-3):
        log.log("CNN Model Initilaizing...")
        if input_shape!=None:
            self.kernel_n=kernel_n
            self.input_shape=input_shape 
            self.stride=stride      # skips, kernel makes at every convolution
            self.dilation=dilation  # kernel coverage
            self.pooling_size=pooling_size
            self.dropout_p=dropout_p
            self.n_output=n_output
            self.learning_rate=learning_rate
            self.model.add(ConvLSTM2D(filters=64,kernel_size=(self.kernel_n,self.kernel_n),activation='relu',
            padding='same',input_shape=self.input_shape,strides=self.stride, dilation_rate=self.dilation, 
            dropout=self.dropout_p))
            self.model.add(Flatten())
            self.model.add(Dense(units=self.input_shape[0] * 32,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=16,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=8,activation='relu'))
            self.model.add(Dropout(self.dropout_p))
            self.model.add(Dense(units=self.n_output))
            #print(self.model.summary())
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),loss=MeanAbsoluteError(),metrics=[MeanSquaredError()])
        log.log("CNN Model Initialized!")        
    def save_model(self,fname):
        self.model.save(fname)
        log.log(f"  Model saved in {fname} file.")
    def load_model(self,fname):
        assert exists(fname)==True , f"{fname} model file does not exists!"
        self.model=load_model(fname)
        log.log(f"  Model loaded from {fname} file.")
    def fit(self,X,Y,batch_size,epochs,validation_split=0.2):
        log.log("  Model training started.")
        self.history=self.model.fit(X,Y,batch_size=batch_size,epochs=epochs,validation_split=validation_split,verbose=0)
        log.log("  Model training completed.")
    def predict(self,X):
        log.log("  Model completed the prediction.")
        return self.model.predict(X)
    def performance(self,Y1,Y2,precision=4):
        assert Y1.shape==Y2.shape , "Shape of both the parameters should be same."
        sum=0
        for i in Y1[0]:
            for j in Y2[0]:
                #sum+=(i-j)**2  
                sum+=abs(i-j)   
        return round(sum/Y1.shape[0],precision)

class Data:
    def __init__(self,filename):
        self.filename=filename
        self.read_file()
        self.plot_location="static/plots/"
    def get_extension(self):
        self.ext=Path(self.filename).suffix.split('.')[-1]
    def read_file(self):
        self.get_extension()
        if self.ext == 'csv':
            self.dataset=read_csv(self.filename)
        elif self.ext== 'xslx':
            self.dataset=read_excel(self.filename)
        else:
            print(f"Data format {self.ext} not supported!")
    def get_columns(self):
        result=[]
        for col in self.dataset.columns:
            result.append(col)
        return result
    def metadata(self,mode):
        # Helper to fetch the metadata of the dataframe which is provided as argument.
        # mode is argument for getting specific metadata printed about the supplied dataframe.
        # 'dataset' : pandas dataframe with column headers set.
        # 'mode' : Integer belonging to the discrete set of [0,1].
        # if mode=0, fetch the columnwise datatype of the datapoints.
        # if mode=1, fetch the columnwise NAN count.
        assert mode==1 or mode==0 , "Mode Should be in the interval [0,1]"
        if mode==0:
            # mode to check the datatypes of each individual columns.
            types={}
            print("Column Name",end="")
            for i in range(35-len("Column Name")):
                print(end=" ")
            print(" Data Type")
            print("-------------------------------------------------------------")
            for col in self.dataset.columns:
                print(col,end="")
                for i in range(30-len(col)):
                    print(end=" ")
                print("|   ",self.dataset[str(col)].dtype)
                types[str(col)]=self.dataset[str(col)].dtype
            return types
        else:
            # mode to check the NAN count of each columns. 
            max_NAN=0        # stores the count of Maximum number of NAN that is present in the dataset.
            colname=""       # stores the columns which is our culprit that contains maximum NAN count
            nans={}
            print("Column Name",end="")
            for i in range(35-len("Column Name")):
                print(end=" ")
            print(" NAN Count")
            print("-------------------------------------------------------------")
            for col in self.dataset.columns:
                cnt=self.dataset[str(col)].isnull().sum()
                nans[str(col)]=cnt
                print(col,end="")
                for i in range(30-len(col)):
                    print(end=" ")
                print("|   ",cnt)
                if cnt>max_NAN:
                    max_NAN=cnt
                    colname=col
            if max_NAN!=0:
                print()
                print(f"Maximum NAN count found is {max_NAN} in column \"{colname}\"! Which is {round(max_NAN/len(self.dataset),4)*100}% of the total dataset.")
            return nans
    def statistical_information(self):
        # Helper function to calculate total row count, Mean , Minimum and Maximum values of each columns data.
        print("Column Name",end="")
        for i in range(35-len("Column Name")):
            print(end=" ")
        print(" Count",end="")
        for i in range(15-len("Count")):
            print(end=" ")
        print(" Mean",end="")
        for i in range(25-len("Mean")):
            print(end=" ")
        print(" Minimum",end="")
        for i in range(25-len("Minumum")):
            print(end=" ")
        print(" Maximum")
        print("--------------------------------------------------------------------------------------------------------------")
        for col in self.dataset.columns:
            if type(self.dataset[str(col)][0]) != type(""):
                print(col,end="")
                for i in range(30-len(col)):
                    print(end=" ")
                print("|   ",self.dataset[str(col)].count(),end="")
                for i in range(10-len(str(self.dataset[str(col)].count()))):
                    print(end=" ")
                print("|   ",round(self.dataset[str(col)].mean(),3),end="")
                for i in range(20-len(str(round(self.dataset[str(col)].mean(),3)))):
                    print(end=" ")
                print("|   ",self.dataset[str(col)].min(),end="")
                for i in range(20-len(str(self.dataset[str(col)].min()))):
                    print(end=" ")
                print("|   ",self.dataset[str(col)].max())
    def denoise(self,mode):
        if mode==0:
            # fill the NANs with the mean
            values=self.dataset.mean(skipna=True).to_dict()    
            self.dataset=self.dataset.fillna(value=values)
        elif mode==1:
            # fill the NANs with the median
            values=self.dataset.median(skipna=True).to_dict()
            self.dataset=self.dataset.fillna(value=values)
        elif mode==2:
            # drop NANs rows.
            self.dataset=self.dataset.dropna()
            self.dataset=self.dataset.reset_index(drop=True)
    def check_cols(self,cols):
        for i in cols:
            if i not in self.dataset.columns:
                return False
        return True 
    def vectorize(self,predictor_cols,target_cols):
        X=[]
        Y=[]
        temp=[]
        predictor_cols=array(predictor_cols)
        target_cols=array(target_cols)
        self.tokens={}
        for col in predictor_cols:
            if self.dataset[str(col)].dtype == type("") and 'date' not in col.lower():
                distinct=[]
                values=self.dataset[str(col)].values
                for value in values:
                    if value not in distinct:
                        distinct.append(value)
                self.tokens[str(col)]={distinct[i] : i for i in range(len(distinct))}
                for i in range(len(values)):
                    values[i]=self.tokens[str(col)][values[i]]
                temp.append(values)
            elif 'date' in col.lower():
                year=[]
                month=[]
                day=[]
                values=self.dataset[str(col)].values
                for value in values:
                    year.append(int(value.split('-')[0]))
                    month.append(int(value.split('-')[1]))
                    day.append(int(value.split('-')[2]))
                temp.append(year)
                temp.append(month)
                temp.append(day)
            else:
                temp.append(self.dataset[str(col)].values)
        temp=array(temp)
        for i in range(temp.shape[1]):
            dims=[]
            for j in range(temp.shape[0]):
                dims.append(temp[j][i])
            X.append(dims)
        temp=[]
        for col in target_cols:
            assert self.dataset[str(col)].dtype != type("") , "Exception : Target column cannot be of String type."
            temp.append(self.dataset[str(col)].values)
        temp=array(temp)
        for i in range(temp.shape[1]):
            dims=[]
            for j in range(temp.shape[0]):
                dims.append(temp[j][i])
            Y.append(dims)
        return X,Y
    def plot(self,mode=0,name='default_plot',val_range=(0,100),area=5,color='green',alpha=0.5):
        fnames=[]
        for col1 in self.dataset.columns:
            for col2 in self.dataset.columns:
                if(col1!=col2 and (type(self.dataset[str(col1)][0]) != type("")) and (type(self.dataset[str(col2)][0]) != type("")) ):
                    if mode==0:
                        # scatter plot
                        #plt.clf()
                        plt.xlabel(f"{col1}")
                        plt.ylabel(f"{col2}")
                        plt.title(f"{col1} versus {col2} Scatter Plot")
                        plt.scatter(self.dataset[str(col1)].values[val_range[0]:val_range[1]],self.dataset[str(col2)].values[val_range[0]:val_range[1]],s=area,c=color,alpha=alpha)
                        plt.savefig(f"{self.plot_location}{name}_{col1}_{col2}.png")
                        plt.close()
                        fnames.append(f"{self.plot_location}{name}_{col1}_{col2}.png")
                    elif mode==1:
                        # line plot
                        #plt.clf()
                        plt.xlabel(f"{col1}")
                        plt.ylabel(f"{col2}")
                        plt.title(f"{col1} versus {col2} Line Plot")
                        plt.plot(self.dataset[str(col1)].values[val_range[0]:val_range[1]],self.dataset[str(col2)].values[val_range[0]:val_range[1]],c=color,alpha=alpha)
                        plt.savefig(f"{self.plot_location}{name}_{col1}_{col2}.png")
                        plt.close()
                        fnames.append(f"{self.plot_location}{name}_{col1}_{col2}.png")
                    else:
                        print("Problem")
        return fnames
    def split(self,X,Y,test_ratio=0.2,shuffle=True,random_state=42):
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_ratio,shuffle=shuffle,random_state=random_state)
        return X_train, X_test, Y_train, Y_test
    def plot_in_time(self,X1,X2,title,label,legend,fname):
        plt.plot(X1)
        plt.plot(X2)
        plt.title(title)
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.legend(legend,loc="upper left")
        plt.savefig(self.plot_location+fname)
        plt.close()
