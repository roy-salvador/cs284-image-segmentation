#############################################################
#
# Demo of Image Segmentation on  Weizmann Horse Database
# @author: Roy Salvador
#
#############################################################

import numpy
import cv2
import time
from os import walk
import imagesegmentation
from os import path
import sys
import ntpath

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
from skimage.transform import resize

from skimage.filters import threshold_otsu

# Pixel Classes - Black, White
horsePixelClasses = [ numpy.array([ 0.,  0.,  0.]), numpy.array([ 1.,  1.,  1.]) ]

# Load trained Model
horseModelLogger = SaveLogger('save/imagesegmentation-horse-hog_96_lbp.model', save_every=1)
horseCRF = horseModelLogger.load()

######################################
# Compute S_0 score
######################################
def foregroundQualityScore(a , b) :
    TP = TN = FP = FN = 0.0
    
    for i in range(0, len(a)) :
        if a[i] == b[i] :
            if a[i] == 0 :
                TN += 1
            else :
                TP += 1
        else :
            if a[i] == 0 :
                FP += 1
            else :
                FN += 1
    
    #print 'accuracy:' + str(((TP+TN) / (TP+FP+FN+TN)))
    #print 'precision:' + str((TP / (TP+FP)))
    #print 'recall:' + str((TP / (TP+FN)))
    #print 'so:' + str(TP / (TP+FP+FN))
    
    return (TP / (TP+FP+FN))
# 
def showSegmentation(rgbfile, labelfile, pixelClasses = horsePixelClasses, crfmodel = horseCRF, visualizeSegmentation=True) :
    start_time = time.time()
    
    # Read RGB and label image
    image = img_as_float(skimageIO.imread(rgbfile))
    bgrImage = cv2.imread(rgbfile,cv2.IMREAD_COLOR)
    #bgrImage = cv2.fastNlMeansDenoisingColored(bgrImage)
    #bgrImage = exposure.adjust_sigmoid(bgrImage)
    if len(image.shape) == 2 : 
                image = color.gray2rgb(image)
                
    # Resize
    #image = resize(image,(120,120), preserve_range=True )
    #bgrImage = resize(bgrImage,(120,120), preserve_range=True)
    
                
    # Derive superpixels and get their average RGB component
    segments = slic(image, n_segments =500, sigma = 1.0)
    rgb_segments = img_as_ubyte(mark_boundaries(image, segments))
    avg_rgb = color.label2rgb(segments, image, kind='avg') 
                
    if labelfile is not None :            
        labelImage = img_as_float(skimageIO.imread(labelfile))
        #labelImage = resize(labelImage,(120,120), preserve_range=True)
        #thresh = threshold_otsu(image)
        #labelImage = labelImage > thresh
        if len(labelImage.shape) == 2 : 
            labelImageRGB = color.gray2rgb(labelImage)
        else :
            labelImageRGB = labelImage
        label_segments = img_as_ubyte(mark_boundaries(labelImageRGB, segments))            
        avg_label = color.label2rgb(segments, labelImageRGB, kind='avg')
     
  

    # Create graph of superpixels and compute their centers
    vertices, edges = imagesegmentation.make_graph(segments)
    gridx, gridy = numpy.mgrid[:segments.shape[0], :segments.shape[1]]
    centers = dict()
    for v in vertices:
        centers[v] = [gridy[segments == v].mean(), gridx[segments == v].mean()]
        
    # compute for features
    xInstance = []
    yInstance = []
    n_features = []            
    n_labels = []
    edge_features = []
    
    
    for v in vertices:
        # unary feature - average rgb of superpixel
        avg_rgb2 = avg_rgb[int(centers[v][1])][int(centers[v][0])]
        hist, hogFeatures = imagesegmentation.getHistogramFeatures(bgrImage, int(centers[v][1]), int(centers[v][0]), forUnaryFeature=True)
        node_feature = numpy.concatenate([avg_rgb2, hist, hogFeatures])
        n_features.append(node_feature)
        
        # label 
        if labelfile is not None :        
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
        # pairwise feature - euclidean distance of adjacent superpixels
        dist = numpy.linalg.norm(avg_rgb[int(centers[e[0]][1])][int(centers[e[0]][0])] - avg_rgb[int(centers[e[1]][1])][int(centers[e[1]][0])] )
        
        
        if e[0] not in histogramCache :
            hist1, lbphist1 = imagesegmentation.getHistogramFeatures(bgrImage, int(centers[e[0]][1]), int(centers[e[0]][0]))
            histogramCache[e[0]] = {'hist' : hist1, 'lbphist' : lbphist1}
        else :
            hist1 =  histogramCache[e[0]]['hist']
            lbphist1 =  histogramCache[e[0]]['lbphist']
        if e[1] not in histogramCache :
            hist2, lbphist2 = imagesegmentation.getHistogramFeatures(bgrImage, int(centers[e[1]][1]), int(centers[e[1]][0]))
            histogramCache[e[1]] = {'hist' : hist2, 'lbphist' : lbphist2}
        else :
            hist2 =  histogramCache[e[1]]['hist']
            lbphist2 =  histogramCache[e[1]]['lbphist']
            
      
        histogramDist = cv2.compareHist(hist1, hist2, 3 )   # Bhattacharyya distance
        textureSimilarity = imagesegmentation.kullback_leibler_divergence(lbphist1, lbphist2) # KL divergence
        
        pairwise_feature = numpy.array([dist, histogramDist, textureSimilarity])
        edge_features.append(pairwise_feature)
    
    
    xInstance.append((numpy.array(n_features), numpy.array(edges), numpy.array(edge_features))) 
    yInstance.append(numpy.array(n_labels))
    
    # Create superpixeled image
    if labelfile is not None :    
        labeledSuperPixeledRGB = numpy.zeros(labelImageRGB.shape)
        labeledSuperPixeled = numpy.zeros(labelImage.shape)
        #print labeledSuperPixeled.shape
        #print labelImageRGB.shape
        for i in range(0,labeledSuperPixeledRGB.shape[0]) :
            for j in range(0,labeledSuperPixeledRGB.shape[1]) :    
                labeledSuperPixeledRGB[i][j] = pixelClasses[n_labels[segments[i][j]]]
                labeledSuperPixeled[i][j] = n_labels[segments[i][j]]
            
    
    # Predict with CRF and build image label
    y_pred = crfmodel.predict(numpy.array(xInstance))
    labeledPredictionRGB = numpy.zeros(image.shape)
    labeledPrediction = numpy.zeros((image.shape[0], image.shape[1]))
    #print labeledPrediction.shape
    for i in range(0,labeledPredictionRGB.shape[0]) :
        for j in range(0,labeledPredictionRGB.shape[1]) :
            labeledPredictionRGB[i][j] = pixelClasses[y_pred[0][segments[i][j]]]
            labeledPrediction[i][j] = y_pred[0][segments[i][j]]
    

    # Print performance
    if labelfile is not None : 
        pixelwise_accuracy = accuracy_score(labelImage.flatten().flatten(),  labeledPrediction.flatten().flatten()) 
        pixelwise_precision = precision_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())
        pixelwise_recall = recall_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())
        pixelwise_f1 = f1_score(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten()) 
        pixelwise_so = foregroundQualityScore(labelImage.flatten().flatten(), labeledPrediction.flatten().flatten())
        
        # comment on f1 score
        if pixelwise_f1 >= 0.9 :
            comment = ' <------------ HIGH!'
        elif pixelwise_f1 <= 0.8 :
            comment = ' <------------ LOW!' 
        else :
            comment = ''
        
        print ''
        print 'Segmentation completed in ' + str(time.time() - start_time) + ' seconds.'
        print 'Total Pixels: ' + str(labelImage.flatten().flatten().shape[0])
        print 'SLIC Pixelwise Accuracy: ' + str(accuracy_score(labelImage.flatten().flatten(), labeledSuperPixeled.flatten().flatten()))
        print ''
        print 'Pixelwise Accuracy: ' + str( pixelwise_accuracy )
        print 'Pixelwise Precision: ' + str( pixelwise_precision )
        print 'Pixelwise Recall: ' + str( pixelwise_recall )
        print 'Pixelwise F1: ' + str( pixelwise_f1 ) + comment
        print 'Pixelwise S0: ' + str( pixelwise_so )
    else :
        'There is no label image hence no performance stats...'

    
    # Show the Images
    if visualizeSegmentation :
        fig, ax = plt.subplots(2, 3)
        fig.canvas.set_window_title('Image Segmentation')
        ax[0, 0].imshow(image)
        ax[0, 0].set_title("Original Image")
        
        ax[0, 1].imshow(rgb_segments)
        ax[0, 1].set_title("Super Pixels")
        
        if labelfile is not None :
            ax[0, 2].imshow(label_segments)
            ax[1, 0].imshow(labelImageRGB)
            ax[1, 1].imshow(labeledSuperPixeledRGB)
            
            
        ax[0, 2].set_title("Segmented Ground Truth")      
        ax[1, 0].set_title("Ground truth")      
        ax[1, 1].set_title("Labeled Super Pixels")
        
        ax[1, 2].imshow(labeledPredictionRGB)
        ax[1, 2].set_title("Prediction")
        
        for a in ax.ravel():
            a.set_xticks(())
            a.set_yticks(())
        plt.show()
        
    # Save result
    #skimageIO.imsave('slic-' + ntpath.basename(rgbfile) , rgb_segments)
    #skimageIO.imsave('slicground-' + ntpath.basename(rgbfile) , label_segments)
    #skimageIO.imsave('result-' + ntpath.basename(rgbfile) , labeledPredictionRGB)
    
    # Return metrics
    if labelfile is not None :
        return pixelwise_accuracy, pixelwise_precision, pixelwise_recall, pixelwise_f1, pixelwise_so
    else :
        return


# Main entry of program
if __name__ == "__main__":

    if len(sys.argv) <= 1 :
        print 'Input horse image filename'
    else :
        rgbFile = sys.argv[1]
        labelFile = rgbFile.replace('image', 'mask')
        
        # Check existence of files
        if not path.isfile(rgbFile) :
            print 'File not found ' + rgbFile
            exit()
        elif not path.isfile(labelFile) or (labelFile == rgbFile):
            print 'Corresponding image file not found. The original image should be named with image-xx and corresponding label image with mask-xx.'
            print 'Proceeding without label'
            showSegmentation(rgbfile = rgbFile, labelfile = None)
        else :
            showSegmentation(rgbfile = rgbFile, labelfile = labelFile)
            #showSegmentation(rgbfile = 'WeizmannSingleScale/horses/training/images/image-105.png', labelfile ='WeizmannSingleScale/horses/training/masks/mask-105.png')


# Good Segmentation in Test Dataset >90% F1 score
# image-108.png
# image-118.png #
# image-12.png #
# image-124.png #
# image-130.png
# image-14.png
# image-142.png
# image-158.png #
# image-164.png
# image-190.png
# image-202.png
# image-210.png
# image-236.png
# image-246.png
# image-262.png
# image-270.png
# image-278.png
# image-304.png
# image-314.png #
# image-320.png
# image-36.png
# image-42.png #
# image-46.png
# image-74.png #
# image-78.png #
# image-80.png


# Poor segmentation in training set <70% F1 score
# image-223.png #
# image-291.png
