import utility

# to generate report of the script output. Redirect the output into a file.

def driver(directory="./static/processing_dir/"):
    data_list=[]
    for i in utility.listdir(directory):
        data_list.append(utility.Data(directory+i))
        print(f"Read dataset file {i} into the analysis pool.")
    for i in data_list:
        print(f"------------ Analysing {i.filename} ---------------")
        print()
        print(f"Following are various columns present in the dataset {i.filename}:")
        cols=i.get_columns()
        for j in range(len(cols)-2):
            print(cols[j], end=", ")
        print(f"{cols[-2]} and {cols[-1]}")
        print()
        print("Let's look at the metadata of the dataset.")
        types=i.metadata(mode=0)
        print()
        nans=i.metadata(mode=1)
        denoise_mode=0
        for j in nans.keys():
            if types[j] == type("") and nans[j]!=0:
                denoise_mode=2
            else:
                denoise_mode=1       # for median replacement
                #denoise_mode=0      # for mean replacement
        i.denoise(denoise_mode)
        print("Statistical information of the dataset is as follows: ")
        i.statistical_information()
        print()
        print("Let's visualize the dataset.")
        i.plot(name=i.filename.split('/')[-1])
        print(f"Plots created and stored at {i.plot_location}")
        print()
        print("Now we are moving ahead in out predictive analysis.")
        print()
        predictor=input("Enter space seperated column names for Predictor set: ").split(' ')
        while not i.check_cols(predictor):
            print("Please enter valid column names.")
            predictor=input("Enter space seperated column names for Predictor set: ").split(' ')

        target=input("Enter space seperated column names for Target set: ").split(' ')
        while not i.check_cols(target):
            print("Please enter valid column names.")
            target=input("Enter space seperated column names for Target set: ").split(' ')
        X,Y=i.vectorize(predictor,target)
        X=utility.array(X)
        X=X.reshape((X.shape[0],1,1,X.shape[1]))
        Y=utility.array(Y)
        model_cnn=utility.Model()
        model_cnn.CNN(input_shape=X.shape[1:],n_output=Y.shape[1])
        X_train, X_test, Y_train, Y_test=i.split(X,Y)
        model_cnn.fit(X_train,Y_train,batch_size=4,epochs=15,validation_split=0.2)
        i.plot_in_time(model_cnn.history.history['loss'],
        model_cnn.history.history['val_loss'],
        "Loss", ["Metrics","Epochs"],["Train","Validation"],f"Loss_cnn_{i.filename.split('/')[-1]}.png")
        i.plot_in_time(model_cnn.history.history['mean_squared_error'],
        model_cnn.history.history['val_mean_squared_error'],
        "MSE", ["Metrics","Epochs"],["Train","Validation"],f"mse_cnn_{i.filename.split('/')[-1]}.png")
        prediction=model_cnn.predict(X_test)
        print(f"MSE for CNN model : {model_cnn.performance(prediction,Y_test)}")
        print()
        X=X.reshape((X.shape[0],1,X.shape[-1]))
        model_lstm=utility.Model()
        model_lstm.LSTM(input_shape=X.shape[1:],n_output=Y.shape[1])
        X_train, X_test, Y_train, Y_test=i.split(X,Y)
        model_lstm.fit(X_train,Y_train,batch_size=4,epochs=15,validation_split=0.2)
        i.plot_in_time(model_lstm.history.history['loss'],
        model_lstm.history.history['val_loss'],
        "Loss", ["Metrics","Epochs"],["Train","Validation"],f"Loss_lstm{i.filename.split('/')[-1]}.png")
        i.plot_in_time(model_lstm.history.history['mean_squared_error'],
        model_lstm.history.history['val_mean_squared_error'],
        "MSE", ["Metrics","Epochs"],["Train","Validation"],f"mse_lstm{i.filename.split('/')[-1]}.png")
        prediction=model_lstm.predict(X_test)
        print(f"MSE for LSTM model : {model_lstm.performance(prediction,Y_test)}")
        utility.rename(i,f"./static/done/{i.filename.split('/')[-1]}")
        print(f"------------------- Analysis of {i.filename.split('/')[-1]} done --------------------")
        print()

if __name__=='__main__':
    driver()







    
