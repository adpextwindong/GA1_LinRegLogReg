import sklearn as sk
import numpy as np



if __name__ == '__main__':
    csv_header = []
    for label in ['time','glucose','slope','iob','mob','morning','afternoon','evening','night']:
        for i in range(7):
            csv_header.append(
                't' + str(i) + '_' + label
            )

    print (csv_header)

def predict_ints(model, dataset, percentile=95):
        err_down = []
        err_up = []
        for x in range(len(dataset)):
            preds=[]
            for pred in model.estimators_:
                preds.append(pred.predict(dataset[x])[0])
                err_down.append(np.percentile(preds, (100 - percentile) / 2.))
                err_up.append(np.percentile(preds, 100 - (100-percentile) / 2.))
                return err_down, err_up



    #dataset = x and y??
     

    #split up dataset here
    #use 1000 randomized trees for some variance

rf = sk.ensemble.RandomForestRegressor(n_estimators=58, criterion='mse', max_depth='none', bootstrap='true')
    #build the regression tree
rf.fit(trainingDataX, trainingDataY)

    

   #lr = sk.LogisticRegression()
    #lr.fit(trainingDataX, trainingDataY)
    #run regression on the classifier
    #svc_w_linear_kernel = SVC(kernel='linear')
trainsize= 58 #leave out 5 for validation
err_down, err_up = predict_ints(rf, dataset[idx[trainsize:]], percentile=90)
truth = datasetY[idx[trainsize:]]
correct = 0.
for i, val in enumerate(truth):
    if err_down[i] <= val <= err_up[i]:
        correct += 1
print correct/len(truth)