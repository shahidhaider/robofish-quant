
import numpy as np




def jaccard_index(pred,truth, threshold):
    threshed_pred = np.where(pred>threshold,1,0)
    intersection = np.multiply(threshed_pred,truth)
    union = threshed_pred+truth
    union = np.where(union>1,1,union)
    
    return intersection.sum()/union.sum()


def accuracy(pred,truth,threshold):
    threshed_pred = np.where(pred>threshold,1,0)
    intersect = np.multiply(threshed_pred,truth)
    return intersect.sum()/truth.sum()


