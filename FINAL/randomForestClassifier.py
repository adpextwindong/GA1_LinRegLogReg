#import sklearn as sk



if __name__ == '__main__':
    csv_header = []
    for label in ['time','glucose','slope','iob','mob','morning','afternoon','evening','night']:
        for i in range(7):
            csv_header.append(
                't' + str(i) + '_' + label
            )

    print (csv_header)

def train_model():

    dataset = csv_header
    #resultDataset 

    #split up dataset here?
    rf = sk.RandomForestClassifier()
    rf.fit(trainX, trainY)

    svc_w_linear_kernel = SVC(kernel='linear')