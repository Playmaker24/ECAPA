'''
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
'''

import os, numpy, torch
import matplotlib.pyplot as plt
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

def init_args(args):
	args.score_save_path    = os.path.join(args.save_path, 'score.txt')
	args.model_save_path    = os.path.join(args.save_path, 'model')
	os.makedirs(args.model_save_path, exist_ok = True)
	return args

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
	
    print("SCORES:", scores)
    print("LABELS:", labels)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)##as in slide the fpr is the FAR(False Acceptance Rate)
    #to create AUC for ROC
    roc_auc = metrics.auc(fpr, tpr)
    print("fpr:", fpr)
    print("tpr:", tpr)
    print("threshold:", thresholds)
    print("successfully go through roc_auc")

    fnr = 1 - tpr ## as in slide fnr is the FRR(False Rejection Rate)
    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
	
    return tunedThreshold, eer, fpr, fnr, tpr, roc_auc

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def accuracy(output, target, topk=(1,)):

	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res

#def plot_confusion_matrix(true_label, pred_label, class_names, title="Confusion Matrix"):
def plot_confusion_matrix(file, fold):
    title = None
    model = None
    true_label = []
    pred_label = []
    save_path = new_path = file.rsplit(f"output_fold_{fold}.txt", 1)[0]
    class_names = ['Negative', 'Positive']
    with open (file, 'r') as f: 
        model = f.readline().strip()
        next(f)
        for line in f:
            tl, pl, _ = line.strip().split(',')
            true_label.append(tl)
            pred_label.append(pl)
    title = 'Confusion Matrix of ' + model
    cm = confusion_matrix(true_label, pred_label)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    #plt.show()
    plt.savefig(save_path + 'cm.png')
    plt.close()

#def plot_roc_curve(fpr, tpr, roc_auc):
def plot_roc_curve(tpr_fpr_file, roc_auc, fold):
    title = None
    model = None
    tpr = []
    fpr = []
    save_path = new_path = tpr_fpr_file.rsplit(f"tpr_fpr_output_fold_{fold}.txt", 1)[0]
    with open (tpr_fpr_file, 'r') as f:
        model = f.readline().strip()
        for line in f:
            fp, tp = line.strip().split(',')
            fpr.append(float(fp))
            tpr.append(float(tp))
    title = 'Receiver Operating Characteristic (ROC) Curve of ' + model
    # Plot the ROC curve
    plt.figure(figsize=(12, 7))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(save_path + 'roc.png')
    plt.close()