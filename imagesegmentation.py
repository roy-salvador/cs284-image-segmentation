#################################################
#
# Trains a CRF model for image segmentation
# @author : Roy Salvador
#
#################################################

import numpy
import cv2
import time
from os import walk


from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import color
from skimage.util import img_as_float
from skimage import io as skimageIO
from skimage import exposure

from pystruct.models import LatentGridCRF, GridCRF, LatentGraphCRF, GraphCRF, EdgeFeatureGraphCRF
from pystruct.learners import LatentSSVM, OneSlackSSVM, SubgradientSSVM, FrankWolfeSSVM
from pystruct.utils import make_grid_edges, SaveLogger
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import skimage.feature as skf
from skimage.exposure import rescale_intensity
from skimage.measure import compare_mse
from skimage import exposure
from skimage.transform import resize


pixelClasses = [ numpy.array([ 0.,  0.,  0.]), numpy.array([ 1.,  1.,  1.]) ]
cnnModel = None
patchSize = 96
patchHalfSize = patchSize/2
cnnPredictFunction = None

segments = None


########################################################################################################
# Generate graph out of SLIC segments
# Taken from http://peekaboo-vision.blogspot.de/2011/08/region-connectivity-graphs-in-python.html
########################################################################################################
def make_graph(grid):
    # get unique labels
    vertices = numpy.unique(grid)
 
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices,numpy.arange(len(vertices))))
    grid = numpy.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
   
    # create edges
    down = numpy.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = numpy.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = numpy.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = numpy.sort(all_edges,axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = numpy.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x%num_vertices],
              vertices[x/num_vertices]] for x in edges] 
 
    return vertices, edges 

##########################################################
# KL Divergence
# Code snippet taken from: http://scikit-image.org/docs/dev/auto_examples/plot_local_binary_pattern.html
##########################################################    
def kullback_leibler_divergence(p, q):
    p = numpy.asarray(p)
    q = numpy.asarray(q)
    filt = numpy.logical_and(p != 0, q != 0)
    return numpy.sum(p[filt] * numpy.log2(p[filt] / q[filt]))
    
    
##################################################################
# Compute the chi-squared distance 
# Code snippet taken from: http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
##################################################################
def chi2_distance(histA, histB, eps = 1e-10):
	d = 0.5 * numpy.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	return d


#########################################################################################################
# Returns Resnet Feature vector
#########################################################################################################    
def getHistogramFeatures(bgrImage, centerY, centerX, forUnaryFeature=False) :
    #print bgrImage.shape
    #print str(centerY) + ' '  + str(centerX)
    patch = numpy.zeros((patchSize,patchSize,3), dtype=numpy.float32)
    
    # construct patch
    for i in range (centerY - patchHalfSize , centerY + patchHalfSize ) :
        for j in range (centerX - patchHalfSize , centerX + patchHalfSize ) :
            if i >=0 and j >= 0 and i<bgrImage.shape[0] and j<bgrImage.shape[1] :
               
                patch[i - (centerY - patchHalfSize)][j - (centerX - patchHalfSize)] = bgrImage[i][j]
    
    # Histogram of intensity values
    hist = cv2.calcHist(patch, [0, 1, 2], None, [12, 12, 12], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, cv2.NORM_L2).flatten()
    
    grayPatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
    if forUnaryFeature :  
        # Histogram of Oriented Gradients
        hog = skf.hog(grayPatch, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=False, transform_sqrt=True)        
        return hist, hog
    else :
        # Histogram of Local Binary Pattern
        lbp = skf.local_binary_pattern(grayPatch, 24, 3, method='nri_uniform')
        n_bins = lbp.max() + 1
        lbphist, _ = numpy.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))        
        return hist, lbphist
    
    
    
#########################################################################################################
# Prepares the Dataset
#########################################################################################################   
def prepareDataset(basedir = 'WeizmannSingleScale/horses/training/images/', labeldir ='WeizmannSingleScale/horses/training/masks/') :

    global pixelClasses
    dataSetX = []
    dataSetX_layer2 =  []
    dataSetY = []
    datasetGroundTruth = []
    

    for (dirpath, dirnames, filenames) in walk(basedir):
        n=0
        for imageFilename in filenames :
            
            #if n>=1:
            #    break
            print imageFilename
            
            n = n+1
        
            # Read RGB and label image
            image = img_as_float(skimageIO.imread(basedir + imageFilename))
            bgrImage = cv2.imread(basedir + imageFilename,cv2.IMREAD_COLOR)
            #bgrImage = cv2.fastNlMeansDenoisingColored(bgrImage)
            #bgrImage = exposure.adjust_sigmoid(bgrImage)
            labelImage = img_as_float(skimageIO.imread(labeldir + imageFilename.replace('image', 'mask') ))
            if len(image.shape) == 2 : 
                image = color.gray2rgb(image)
                
            # Resize
            #image = resize(image,(120,120), preserve_range=True )
            #bgrImage = resize(bgrImage,(120,120), preserve_range=True)
            #labelImage = resize(labelImage,(120,120), preserve_range=True)
            
            
            # Scan label image for additional classes
            if len(labelImage.shape) == 2 : 
                labelImageRGB = color.gray2rgb(labelImage)
            else :
                labelImageRGB = labelImage
            #for i in range(0,labelImageRGB.shape[0]) :
            #    for j in range(0,labelImageRGB.shape[1]) :
            #        if len(pixelClasses) == 0 :
            #            pixelClasses.append(labelImageRGB[i][j])
            #        else :
            #            isAlreadyAPixelClass = False
            #            for pixelClass in pixelClasses:
            #                if numpy.array_equal(pixelClass, labelImageRGB[i][j]) :
            #                    isAlreadyAPixelClass = True
            #                    break
            #            if not isAlreadyAPixelClass :
            #                pixelClasses.append(labelImageRGB[i][j])
            #print pixelClasses
            
            # Derive superpixels and get their average RGB component
            segments = slic(image, n_segments = 500, sigma = 1.0)
            rgb_segments = img_as_ubyte(mark_boundaries(image, segments))
            label_segments = img_as_ubyte(mark_boundaries(labelImageRGB, segments))            
            avg_rgb = color.label2rgb(segments, image, kind='avg') 
            avg_label = color.label2rgb(segments, labelImageRGB, kind='avg') 
            #avg_cie_rgb = color.rgb2lab(avg_rgb)
            #avg_cie_label = color.rgb2lab(avg_label)

            # Create graph of superpixels and compute their centers
            vertices, edges = make_graph(segments)
            gridx, gridy = numpy.mgrid[:segments.shape[0], :segments.shape[1]]
            centers = dict()
            for v in vertices:
                centers[v] = [gridy[segments == v].mean(), gridx[segments == v].mean()]
            #print vertices
            #print edges
            #print centers
            
            # Build training instances
            n_features = []
            edge_features = []          
            n_labels = []
                    
            # Compute image centers        
            centerX = labelImageRGB.shape[1] / 2.0
            centerY = labelImageRGB.shape[0] / 2.0
            
            for v in vertices:
                # unary features layer 1 - average rgb of superpixel, histogram of patch surrounding center and CNN features
                avg_rgb2 = avg_rgb[int(centers[v][1])][int(centers[v][0])]
                hist, hogFeatures = getHistogramFeatures(bgrImage, int(centers[v][1]), int(centers[v][0]), forUnaryFeature=True)
                #relativeX = (centers[v][1] - centerX) / centerX
                #relativeY = (centers[v][0] - centerY) / centerY
                node_feature = numpy.concatenate([avg_rgb2, hist, hogFeatures]) 
                n_features.append(node_feature)
                                 
                # label
                minEuclideanDistance = numpy.inf # simulate infinity
                pixelClass = -1 
                for i in range(0, len(pixelClasses)) :
                    # set the label of the superpixel to the pixelClass with minimum euclidean distance
                    dist = numpy.linalg.norm( avg_label[int(centers[v][1])][int(centers[v][0])] - pixelClasses[i] )
                    if dist < minEuclideanDistance :
                        pixelClass = i
                        minEuclideanDistance = dist
                n_labels.append(pixelClass)
            
            histogramCache = {}
            for e in edges :
                # pairwise feature layer 1 - histogram distance, avg RGB euclidean distance of adjacent superpixels , texture similarity
                dist = numpy.linalg.norm(avg_rgb[int(centers[e[0]][1])][int(centers[e[0]][0])] - avg_rgb[int(centers[e[1]][1])][int(centers[e[1]][0])] )
                
                
                if e[0] not in histogramCache :
                    hist1, lbphist1 = getHistogramFeatures(bgrImage, int(centers[e[0]][1]), int(centers[e[0]][0]))
                    histogramCache[e[0]] = {'hist' : hist1, 'lbphist' : lbphist1}
                else :
                    hist1 =  histogramCache[e[0]]['hist']
                    lbphist1 =  histogramCache[e[0]]['lbphist']
                if e[1] not in histogramCache :
                    hist2, lbphist2 = getHistogramFeatures(bgrImage, int(centers[e[1]][1]), int(centers[e[1]][0]))
                    histogramCache[e[1]] = {'hist' : hist2, 'lbphist' : lbphist2}
                else :
                    hist2 =  histogramCache[e[1]]['hist']
                    lbphist2 =  histogramCache[e[1]]['lbphist']
                
                histogramDist = cv2.compareHist(hist1, hist2, 3 )   # Bhattacharyya distance
                textureSimilarity = kullback_leibler_divergence(lbphist1, lbphist2) # KL divergence

                
                
                pairwise_feature = numpy.array([dist, histogramDist, textureSimilarity])
                edge_features.append(pairwise_feature)
                
            
            # Add to dataset
            dataSetX.append((numpy.array(n_features), numpy.array(edges), numpy.array(edge_features)))  
            dataSetY.append(numpy.array(n_labels))
    
    return dataSetX, dataSetY
            
            
            
#########################################################################################################
# Evaluates Superpixelwise Performance of the Dataset
######################################################################################################### 
def evaluatePerformance(clf, datasetX, datasetY) :
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    y_pred = clf.predict(datasetX)
    for i in range(0, len(y_pred)) :
        accuracy += accuracy_score(y_pred[i], datasetY[i])
        precision += precision_score(y_pred[i], datasetY[i])
        recall += recall_score(y_pred[i], datasetY[i])
        f1 += f1_score(y_pred[i], datasetY[i])
        #jaccard += jaccard_similarity_score(y_pred[i], datasetY[i])
        
   
    print 'Superpixelwise Accuracy: ' + str(accuracy / len(y_pred))
    print 'Superpixelwise Precision: ' + str(precision / len(y_pred))
    print 'Superpixelwise Recall: '+ str(recall / len(y_pred))
    print 'Superpixelwise F1: ' + str(f1 / len(y_pred))
    
    return y_pred
            
           
#########################################################################################################
# Train the CRF
#########################################################################################################             
def train(trainSetX, trainSetY, testSetX, testSetY) :
    modelLogger = SaveLogger('imagesegmentation-horse-hog_96_lbp_test.model', save_every=1)
    
    # Load trained CRF model
    print 'Loading trained model for  CRF'
    #clf = modelLogger.load()
    
    # Uncomment if we want to train from scratch first layer CRF
    print 'Training CRF...'
    start_time = time.time()
    crf = EdgeFeatureGraphCRF()#antisymmetric_edge_features=[1,2]
    clf = FrankWolfeSSVM(model=crf, C=10., tol=.1, verbose=3, show_loss_every=1,  logger=modelLogger)  # #max_iter=50
    ##clf = OneSlackSSVM(model=crf, verbose=1, show_loss_every=1, logger=modelLogger)
    clf.fit(numpy.array(trainSetX), numpy.array(trainSetY))
    print 'Training CRF took ' + str(time.time() - start_time) + ' seconds'
    
    #print("Overall super pixelwise accuracy (training set): %f" % clf.score(numpy.array(trainSetX), numpy.array(trainSetY) ))
    #print("Overall super pixelwise accuracy (test set): %f" % clf.score(numpy.array(testSetX), numpy.array(testSetY) ))
    
    print 'SUPERPIXELWISE ACCURACY'
    print '-----------------------------------------------------------------------'
    print ''
    print 'TRAINING SET RESULTS'
    train_ypred = evaluatePerformance(clf, numpy.array(trainSetX), numpy.array(trainSetY))
    print ''
    print 'TEST SET RESULTS'
    evaluatePerformance(clf, numpy.array(testSetX), numpy.array(testSetY))
    print '-----------------------------------------------------------------------'
    


# Main entry of program
if __name__ == "__main__":

    
    print '==================================================='
    print 'Preparing training set'
    trainSetX, trainSetY = prepareDataset(basedir = 'WeizmannSingleScale/horses/training/images/', labeldir ='WeizmannSingleScale/horses/training/masks/')
    print '==================================================='
    print 'Preparing test set'
    testSetX, testSetY = prepareDataset(basedir = 'WeizmannSingleScale/horses/test/images/', labeldir ='WeizmannSingleScale/horses/test/masks/')
    print '==================================================='
    print 'Training the CRF'
    train(trainSetX, trainSetY, testSetX, testSetY)