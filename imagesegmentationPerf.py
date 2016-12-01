################################################################
#
# Measures the global pixelwise performance of the trained CRF
# @author: Roy Salvador
#
###############################################################

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

from os import walk
import imagesegmentationDemo

# Measures dataset performance
def measurePerformance( basedir = 'WeizmannSingleScale/horses/training/images/', labeldir ='WeizmannSingleScale/horses/training/masks/') :
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    s0 = 0.0
    n = 0
    
    print '*****************************************************************************'
    print 'Evaluating dataset at ' + basedir
    print '*****************************************************************************'

    # get metric scores for each image in the dataset
    for (dirpath, dirnames, filenames) in walk(basedir):
        for imageFilename in filenames :
            print '---------------------------------------------------------------------------------------------'
            print imageFilename
            pixelwise_accuracy, pixelwise_precision, pixelwise_recall, pixelwise_f1, pixelwise_s0 =  \
                imagesegmentationDemo.showSegmentation(basedir + imageFilename, labeldir + imageFilename.replace('image', 'mask'), visualizeSegmentation=False)
            
            accuracy += pixelwise_accuracy
            precision += pixelwise_precision
            recall += pixelwise_recall
            f1 += pixelwise_f1
            s0 += pixelwise_s0
            n += 1
            #break
        
    print '***************************************************'
    print 'Average Pixelwise Accuracy: ' + str(accuracy / n)
    print 'Average Pixelwise Precision: ' + str(precision / n)
    print 'Average Pixelwise Recall: '+ str(recall / n)
    print 'Average Pixelwise F1: ' + str(f1 / n) 
    print 'Average Pixelwise S0: ' + str(s0 / n) 
    print '***************************************************'

# Main entry of program
if __name__ == "__main__":
    # Training set
    measurePerformance( basedir = 'WeizmannSingleScale/horses/training/images/', labeldir ='WeizmannSingleScale/horses/training/masks/')
    # Test set
    measurePerformance( basedir = 'WeizmannSingleScale/horses/test/images/', labeldir ='WeizmannSingleScale/horses/test/masks/')
    
    
    
    

    
