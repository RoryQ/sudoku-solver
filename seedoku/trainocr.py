import tinyocr, glob, timeit, sys, os, cPickle

# open std out in unbuffered mode for print statements
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

print "===opening ocr db==="
start = timeit.default_timer()
ocr = tinyocr.OCRManager('ocrdb')
stop = timeit.default_timer()
print stop - start

print "===reading training data==="
start = timeit.default_timer()
for i in xrange(1, 10):
    path = "data/%i/*.png" % i
    for f in glob.glob(path):
        ocr.add(f, str(i))
stop = timeit.default_timer()
print stop - start

print "===computing features==="
start = timeit.default_timer()
ocr.compute_features()
stop = timeit.default_timer()
print stop - start

print "===training learner==="
start = timeit.default_timer()
ocr.train_learner()
stop = timeit.default_timer()
print stop - start

print "===testing ocr==="
for i in xrange(1, 10):
    path = "data/%i/*.png" % i
    for f in glob.glob(path):
        if ocr.test(f)[0] != str(i):
            print "%s expected: %i\tfound: %s" % (f, i, ocr.test(f)[0])

print "===pickling ocr object==="
start = timeit.default_timer()
with open('ocr.pkl', 'wb') as output:
    cPickle.dump(ocr, output, cPickle.HIGHEST_PROTOCOL)
stop = timeit.default_timer()
print stop - start

print "complete."

