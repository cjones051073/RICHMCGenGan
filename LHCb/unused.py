
def createPersistentDatasets( maxData = -1 ):

    import gzip, pickle

    # get the normalised data
    train_input, test_input, train_target, test_target = createNormalisedData(maxData)

    print "Pickling the data"

    # pickle the data
    f = gzip.open("data/train_input.pck.gz","w")
    pickle.dump(train_input,f)
    f.close()
    f = gzip.open("data/test_input.pck.gz","w")
    pickle.dump(test_input,f)
    f.close()
    f = gzip.open("data/train_target.pck.gz","w")
    pickle.dump(train_target,f)
    f.close()
    f = gzip.open("data/test_target.pck.gz","w")
    pickle.dump(test_target,f)
    f.close()

def loadPersistentDatasets():

    print "Loading the pickled the data"

    import gzip, pickle

    # load the data
    f = gzip.open("train_input.pck.gz","r")
    train_input = pickle.load(f)

    return train_input 
