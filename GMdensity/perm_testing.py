from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.spm.preprocess import VBMSegment, Segment
from nipype.interfaces.ants import Atropos, Registration, ApplyTransforms, N4BiasFieldCorrection
from nipype.interfaces.fsl import ApplyMask, BET
from pandas import DataFrame, Series, read_csv

# Study specific variables
study_home = '/moochie/Cat/Aggregate_anats/GMD_ML'

sub_data_file = study_home + '/doc/subject_info.csv'
subject_info = read_csv(sub_data_file, index_col=0)
subjects_list = subject_info['freesurferID'].tolist()

preproc_dir = study_home + '/proc'
output_dir = study_home + '/ml_trainingset'

sample_template = study_home + '/templates/lcbd_template_1mm.nii.gz'
sample_template_brain = study_home + '/templates/lcbd_template_1mm_brain.nii.gz'
sample_template_mask = study_home + '/templates/lcbd_template_1mm_mask.nii.gz'
gmd_feature_data = output_dir + '/gmd_combined.nii.gz'


from sklearn.preprocessing import StandardScaler, PowerTransformer
from numpy import squeeze

## Create a conditions list for the feature set
age_labels = subject_info[['Age_yrs']].copy()
age_labels = age_labels.values
irr_labels = subject_info[['MAP_Temper_Loss','MAP_Noncompliance','MAP_General_Aggression','MAP_Low_Concern']].copy()
irr_labels = irr_labels.values

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(age_labels)
sd_agedata = scaler.transform(age_labels)

pt = PowerTransformer()
pt.fit(irr_labels)
pt_irritability = pt.transform(irr_labels)
pt_irritability = squeeze(pt_irritability)

conditions = DataFrame(data=sd_agedata, index=None, columns=['age'])
conditions['subject'] = Series(subjects_list, index=conditions.index)
conditions = conditions.merge(DataFrame(pt_irritability,
                                        columns=['Temper_Loss','Noncompliance','General_Aggression','Low_Concern'],
                                        index=conditions.index),left_index=True, right_index=True)
conditions['sequence'] = subject_info['Sequence.Version']
conditions['dbfactor'] = subject_info['DB_factor']
conditions['angfactor'] = subject_info['anger_factor']
conditions['age_yrs'] = subject_info['Age_yrs']

from nilearn.input_data import NiftiMasker


masker = NiftiMasker(mask_img=sample_template_mask,standardize=True, 
                     memory='nilearn_cache', memory_level=1)
X = masker.fit_transform(gmd_feature_data)

for analysis in ['age_LOSO','Temper_Loss_LOSO','General_Aggression_LOSO','angfactor_LOSO','Low_Concern_LOSO','Noncompliance_LOSO']:
    if analysis == 'age_LOSO':
        labels = conditions['age']
        groups = conditions['subject']
    elif analysis == 'Temper_Loss_LOSO':
        labels = conditions['Temper_Loss']
        groups = conditions['subject']
    elif analysis == 'Noncompliance_LOSO':
        labels = conditions['Noncompliance']
        groups = conditions['subject']
    elif analysis == 'General_Aggression_LOSO':
        labels = conditions['General_Aggression']
        groups = conditions['subject']
    elif analysis == 'Low_Concern_LOSO':
        labels = conditions['Low_Concern']
        groups = conditions['subject'] 
    elif analysis == 'DBfactor_LOSO':
        labels = conditions['dbfactor']
        groups = conditions['subject'] 
    elif analysis == 'angfactor_LOSO':
        labels = conditions['angfactor']
        groups = conditions['subject'] 
    
    from sklearn.model_selection import permutation_test_score
    import matplotlib.pyplot as plt
    from numpy import savetxt
    from sklearn.feature_selection import f_regression, SelectPercentile
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut

    loso = LeaveOneGroupOut()

    # Set up the regression
    svr = SVR(kernel='linear', C=1)

    feature_selection = SelectPercentile(f_regression, percentile=5)
    fs_svr = Pipeline([('feat_select', feature_selection), ('svr', svr)])


    results_file = open(output_dir + '/perm_results_' + analysis + '.txt','w')

    score, permutation_scores, pvalue = permutation_test_score(fs_svr, X, labels, scoring='neg_mean_squared_error', 
                                                               cv=loso, n_permutations=500, n_jobs=20, 
                                                               groups=groups)
    savetxt(output_dir + '/permutation_scores_mse_' + analysis + '.txt', permutation_scores)

    # Save a figure of the permutation scores
    plt.hist(permutation_scores, 20, label='Permutation scores',
             edgecolor='black')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
             label='Mean Squared Error (pvalue %f)' % pvalue)
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.savefig(output_dir + '/permutation_plot_mse_' + analysis + '.svg', transparent=True)
    plt.close()

    # save final pval/classifier score
    results_file.write('MSE score %s (pvalue : %s) \n' % (score, pvalue))

    ## Perform permutation testing to get a p-value for r-squared
    score, permutation_scores, pvalue = permutation_test_score(fs_svr, X, labels, scoring='r2', 
                                                               cv=loso, n_permutations=500, n_jobs=20, 
                                                               groups=groups)
    savetxt(output_dir + '/permutation_scores_r2_' + analysis + '.txt', permutation_scores)

    # Save a figure of the permutation scores
    plt.hist(permutation_scores, 20, label='Permutation scores',
             edgecolor='black')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
             label='R-squared (pvalue %f)' % pvalue)
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.savefig(output_dir + '/permutation_plot_r2_' + analysis + '.svg', transparent=True)
    plt.close()

    # save final pval/classifier score
    results_file.write('R square: %s (pvalue : %s) \n' % (score, pvalue))
    results_file.close()