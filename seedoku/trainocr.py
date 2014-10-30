import mytinyocr, gzip, numpy, cPickle, timeit, shutil

f = gzip.open('mnist.pkl.gz')
t, v, tt = cPickle.load(f)
f.close()

def reshapelist(l):
    return numpy.array([a.reshape((28, 28)) for a in l])
    
def fixset(l):
    return (reshapelist(l[0]), l[1])

def i2c_list(l):
    return numpy.array([str(a) for a in l])
    
train_set = fixset(t)
valid_set = fixset(v)
test_set = fixset(tt)

ocrdbname = 'ocrdb'

print "adding images to database"
start = timeit.default_timer()
ocr = mytinyocr.OCRManager(ocrdbname)
ocr.add_many(train_set[0], numpy.array([str(a) for a in train_set[1]]))
stop = timeit.default_timer()
print stop - start

print "computing features"
start = timeit.default_timer()
ocr.compute_features()
stop = timeit.default_timer()
print stop - start

print "training learner"
start = timeit.default_timer()
ocr.train_learner()
stop = timeit.default_timer()
print stop - start
