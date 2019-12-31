import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
import seaborn as sns
from sklearn import neighbors, svm, preprocessing, metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.externals.six import StringIO
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import bagging, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.multiclass import OutputCodeClassifier, OneVsOneClassifier, OneVsRestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from IPython.display import Image
from scipy.stats import friedmanchisquare, wilcoxon
from pandas.api.types import is_numeric_dtype

#Codigo p3.py reutilizado
def categorizar(df):
	for i in list(df.columns.values):
		if is_numeric_dtype(df[i]) == False:
			df[i] = df[i].astype('category').cat.codes

def dividir_train_test(dataframe, percentTrain):
	"""
	- Función que divide un Dataframe en dos Dfs, uno con los patrones de Train y otro con los patrones de Test, de forma aleatoria.
	- Argumentos de entrada:
		- dataframe: Dataframe que vamos a utilizar para extraer los datos.
		- percentTrain: Porcentaje de patrones en entrenamiento.
	- Return:
		- train: Df con los datos de train.
		- test: Df con los datos de test.
	"""
	mascara = np.random.rand(len(dataframe)) < percentTrain
	train = dataframe[mascara]
	test = dataframe[~mascara]
	return train, test

#Copypasteado parcialmente de la practica de IMC
def preprocesar_datos(filename):
	label_e = preprocessing.LabelEncoder()
	data=pd.read_csv(filename)
	X = data.iloc[:, :-1].values
	y = data.iloc[:, -1].values
	df=pd.DataFrame(data)
	categorizar(df)
	train, test = dividir_train_test(df, 0.7)
	trainInputs = train.iloc[:, :-1].values
	trainOutputs = train.iloc[:, -1].values
	testInputs = test.iloc[:, :-1].values
	testOutputs = test.iloc[:, -1].values
	# print(trainInputs)
	# print(trainOutputs)
	label_e.fit(trainOutputs)
	trainOutputs=label_e.transform(trainOutputs)
	testOutputs=label_e.transform(testOutputs)
	return X, y, df, trainInputs, trainOutputs, testInputs, testOutputs

def clasificar_dt(X, y, df, trainInputs, trainOutputs, testInputs, testOutputs, graphname, features, classes):
	print("\n[" + str(graphname) + "]")
	clf=DecisionTreeClassifier()
	scores = cross_val_score(clf, X, y, cv=10)
	# Train DT
	clf=clf.fit(trainInputs, trainOutputs)
	# Predict for test
	precisionTrain = clf.score(trainInputs, trainOutputs)
	precisionTest = clf.score(testInputs, testOutputs)
	print("\tCCR train = %.2f%% | CCR test = %.2f%%" % (precisionTrain*100, precisionTest*100))
	prediccion_test = clf.predict(testInputs)
	print(prediccion_test)
	print(testOutputs)
	dot_data = StringIO()
	export_graphviz(clf, out_file = dot_data, filled = True, rounded = True, special_characters = True, feature_names = features, class_names = classes)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_svg(graphname)
	Image(graph.create_svg())
	return precisionTest

#Clasificador one vs one
def clasificar_OVO(X, y, df, trainInputs, trainOutputs, testInputs, testOutputs, graphname):
	print("\n[" + str(graphname) + "]")
	clfBase=DecisionTreeClassifier()
	scores = cross_val_score(clfBase, X, y, cv=10)
	clf=OneVsOneClassifier(clfBase)
	clf=clf.fit(trainInputs, trainOutputs)
	precisionTrain = clf.score(trainInputs, trainOutputs)
	precisionTest = clf.score(testInputs, testOutputs)
	print("\tCCR train = %.2f%% | CCR test = %.2f%%" % (precisionTrain*100, precisionTest*100))
	prediccion_test = clf.predict(testInputs)
	print(prediccion_test)
	print(testOutputs)
	return precisionTest

#Clasificador one vs all
def clasificar_OVA(X, y, df, trainInputs, trainOutputs, testInputs, testOutputs, graphname):
	print("\n[" + str(graphname) + "]")
	clfBase=DecisionTreeClassifier()
	scores = cross_val_score(clfBase, X, y, cv=10)
	clf=OneVsRestClassifier(clfBase)
	clf=clf.fit(trainInputs, trainOutputs)
	precisionTrain = clf.score(trainInputs, trainOutputs)
	precisionTest = clf.score(testInputs, testOutputs)
	print("\tCCR train = %.2f%% | CCR test = %.2f%%" % (precisionTrain*100, precisionTest*100))
	prediccion_test = clf.predict(testInputs)
	print(prediccion_test)
	print(testOutputs)
	return precisionTest

def clasificar_ECOC(X, y, df, trainInputs, trainOutputs, testInputs, testOutputs, graphname):
	print("\n[" + str(graphname) + "]")
	kernelRBF=1.0*RBF(1.0)
	clf=OutputCodeClassifier(estimator = DecisionTreeClassifier())
	clf=clf.fit(trainInputs, trainOutputs)
	precisionTrain = clf.score(trainInputs, trainOutputs)
	precisionTest = clf.score(testInputs, testOutputs)
	print("\tCCR train = %.2f%% | CCR test = %.2f%%" % (precisionTrain*100, precisionTest*100))
	prediccion_test = clf.predict(testInputs)
	print(prediccion_test)
	print(testOutputs)
	return precisionTest

X_iris, y_iris, df_iris, trI_iris, trO_iris, teI_iris, teO_iris = preprocesar_datos('Datasets/iris.csv')

X_contact_lenses, y_contact_lenses, df_contact_lenses, trI_contact_lenses, trO_contact_lenses, teI_contact_lenses, teO_contact_lenses = preprocesar_datos('Datasets/contact_lenses.csv')

X_segment_challenge, y_segment_challenge, df_segment_challenge, trI_segment_challenge, trO_segment_challenge, teI_segment_challenge, teO_segment_challenge = preprocesar_datos('Datasets/segment_challenge.csv')

X_soybean, y_soybean, df_soybean, trI_soybean, trO_soybean, teI_soybean, teO_soybean = preprocesar_datos('Datasets/soybean.csv')

X_segment_test, y_segment_test, df_segment_test, trI_segment_test, trO_segment_test, teI_segment_test, teO_segment_test = preprocesar_datos('Datasets/segment_test.csv')

X_glass, y_glass, df_glass, trI_glass, trO_glass, teI_glass, teO_glass = preprocesar_datos('Datasets/glass.csv')

X_ecoli, y_ecoli, df_ecoli, trI_ecoli, trO_ecoli, teI_ecoli, teO_ecoli = preprocesar_datos('Datasets/ecoli.csv')

X_post_operative, y_post_operative, df_post_operative, trI_post_operative, trO_post_operative, teI_post_operative, teO_post_operative = preprocesar_datos('Datasets/post_operative.csv')

X_car, y_car, df_car, trI_car, trO_car, teI_car, teO_car = preprocesar_datos('Datasets/car.csv')

X_iris_2D, y_iris_2D, df_iris_2D, trI_iris_2D, trO_iris_2D, teI_iris_2D, teO_iris_2D = preprocesar_datos('Datasets/iris_2D.csv')


arrayScores_dt = np.array(
	[
		clasificar_dt(
			X_iris,
			y_iris,
			df_iris,
			trI_iris,
			trO_iris,
			teI_iris,
			teO_iris,
			'P5_iris_dt.svg',
			['sepallength','sepalwidth','petallength','petalwidth'],
			['Iris_setosa', 'Iris_versicolor', 'Iris_virginica']
		),
		clasificar_dt(
			X_contact_lenses,
			y_contact_lenses,
			df_contact_lenses,
			trI_contact_lenses,
			trO_contact_lenses,
			teI_contact_lenses,
			teO_contact_lenses,
			'P5_contact_lenses_dt.svg',
			['young','pre_presbyopic','presbyopic','hypermetrope','astigmatism','normal_tear_prod_rate'],
			['hard', 'soft', 'none']
		),
		clasificar_dt(
			X_segment_challenge,
			y_segment_challenge,
			df_segment_challenge,
			trI_segment_challenge,
			trO_segment_challenge,
			teI_segment_challenge,
			teO_segment_challenge,
			'P5_segment_challenge_dt.svg',
			['region_centroid_col','region_centroid_row','region_pixel_count','short_line_density_5','short_line_density_2','vedge_mean','vegde_sd','hedge_mean','hedge_sd','intensity_mean','rawred_mean','rawblue_mean','rawgreen_mean','exred_mean','exblue_mean','exgreen_mean','value_mean','saturation_mean','hue_mean'],
			['brickface','sky','foliage','cement','window','path','grass']
		),
		clasificar_dt(
			X_soybean,
			y_soybean,
			df_soybean,
			trI_soybean,
			trO_soybean,
			teI_soybean,
			teO_soybean,
			'P5_soybean_dt.svg',
			['april','may','june','july','august','september','october','plant_stand_lt_normal','precip_lt_norm','precip_norm','precip_gt_norm','temp_lt_norm','temp_norm','temp_gt_norm','no_hail','crop_hist_diff_lst_year','crop_hist_same_lst_yr','crop_hist_same_lst_two_yrs','crop_hist_same_lst_sev_yrs','area_damaged_scattered','area_damaged_low_areas','area_damaged_upper_areas','area_damaged_whole_field','severity_minor','severity_pot_severe','severity_severe','seed_tmt_none','seed_tmt_fungicide','seed_tmt_other','germination_90_100','germination_80_89','germination_lt_80','plant_abnorm_growth','abnorm_leaves','leafspots_halo_absent','leafspots_halo_yellow_halos','leafspots_halo_no_yellow_halos','leafspots_marg_w_s_marg','leafspots_marg_no_w_s_marg','leafspots_marg_dna','leafspot_size_lt_1/8','leafspot_size_gt_1/8','leafspot_size_dna','leaf_shread','leaf_malf','leaf_mild_absent','leaf_mild_upper_surf','leaf_mild_lower_surf','abnorm_stem','no_lodging','stem_cankers_absent','stem_cankers_below_soil','stem_cankers_above_soil','stem_cankers_above_sec_nde','canker_lesion_dna','canker_lesion_brown','canker_lesion_dk_brown_blk','canker_lesion_tan','fruiting_bodies','external_decay_absent','external_decay_firm_and_dry','external_decay_watery','mycelium','int_discolor_none','int_discolor_brown','int_discolor_black','sclerotia','fruit_pods_norm','fruit_pods_diseased','fruit_pods_few_present','fruit_pods_dna','fruit_spots_absent','fruit_spots_colored','fruit_spots_brown_w/blk_specks','fruit_spots_distort','fruit_spots_dna','abnorm_seed','mold_growth','seed_discolor','seed_size_lt_norm','shriveling','roots_norm','roots_rotted','roots_galls_cysts'],
			['herbicide_injury', 'cyst_nematode', 'diaporthe_pod_and_stem_blight', '2_4_d_injury', 'diaporthe_stem_canker', 'charcoal_rot', 'rhizoctonia_root_rot', 'powdery_mildew', 'downy_mildew', 'bacterial_blight', 'bacterial_pustule', 'purple_seed_stain', 'phyllosticta_leaf_spot', 'brown_stem_rot', 'anthracnose', 'phytophthora_rot', 'alternarialeaf_spot', 'frog_eye_leaf_spot', 'brown_spot']
		),
		clasificar_dt(
			X_segment_test,
			y_segment_test,
			df_segment_test,
			trI_segment_test,
			trO_segment_test,
			teI_segment_test,
			teO_segment_test,
			'P5_segment_test_dt.svg',
			['region_centroid_col','region_centroid_row','region_pixel_count','short_line_density_5','short_line_density_2','vedge_mean','vegde_sd','hedge_mean','hedge_sd','intensity_mean','rawred_mean','rawblue_mean','rawgreen_mean','exred_mean','exblue_mean','exgreen_mean','value_mean','saturation_mean','hue_mean'],
			['brickface','sky','foliage','cement','window','path','grass']
		),
		clasificar_dt(
			X_glass,
			y_glass,
			df_glass,
			trI_glass,
			trO_glass,
			teI_glass,
			teO_glass,
			'P5_glass_dt.svg',
			['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'],
			['vehic wind non_float', 'tableware', 'containers', 'vehic wind float', 'headlamps', 'build wind float', 'build wind non_float']
		),
		clasificar_dt(
			X_ecoli,
			y_ecoli,
			df_ecoli,
			trI_ecoli,
			trO_ecoli,
			teI_ecoli,
			teO_ecoli,
			'P5_ecoli_dt.svg',
			['mcg','gvh','lip','chg','aac','alm1','alm2'],
			['imS','imL','omL','om','imU','pp','im','cp']
		),
		clasificar_dt(
			X_post_operative,
			y_post_operative,
			df_post_operative,
			trI_post_operative,
			trO_post_operative,
			teI_post_operative,
			teO_post_operative,
			'P5_post_operative_dt.svg',
			['MID_L_CORE','HIGH_L_CORE','LOW_L_CORE','LOW_L_SURF','HIGH_L_SURF','MID_L_SURF','GOOD_L_O2','MID_L_BP','HIGH_L_BP','LOW_L_BP','UNSTABLE_SURF','STABLE_CORE','UNSTABLE_CORE','MOD_STABLE_CORE','STABLE_BP','MOD_STABLE_BP','UNSTABLE_BP','COMFORT'],
			['I','S','A']
		),
		clasificar_dt(
			X_car,
			y_car,
			df_car,
			trI_car,
			trO_car,
			teI_car,
			teO_car,
			'P5_car_dt.svg',
			['buying_vhigh','buying_high','buying_med','buying_low','maint_vhigh','maint_high','maint_med','maint_low','doors_2','doors_3','doors_4','doors_more_or_equal_than_5','persons_2','persons_4','persons_more_than_4','log_boot_small','log_boot_med','log_boot_big','safety_low','safety_med','safety_high'],
			['v_good', 'good', 'acc', 'unacc']
		),
		clasificar_dt(
			X_iris_2D,
			y_iris_2D,
			df_iris_2D,
			trI_iris_2D,
			trO_iris_2D,
			teI_iris_2D,
			teO_iris_2D,
			'P5_iris_2D_dt.svg',
			['petallength','petalwidth'],
			['Iris_setosa', 'Iris_versicolor', 'Iris_virginica']
		)
	]
)

arrayScores_OVO = np.array(
	[
		clasificar_OVO(
			X_iris,
			y_iris,
			df_iris,
			trI_iris,
			trO_iris,
			teI_iris,
			teO_iris,
			'P5_iris_OVO.svg'
		),
		clasificar_OVO(
			X_contact_lenses,
			y_contact_lenses,
			df_contact_lenses,
			trI_contact_lenses,
			trO_contact_lenses,
			teI_contact_lenses,
			teO_contact_lenses,
			'P5_contact_lenses_OVO.svg'
		),
		clasificar_OVO(
			X_segment_challenge,
			y_segment_challenge,
			df_segment_challenge,
			trI_segment_challenge,
			trO_segment_challenge,
			teI_segment_challenge,
			teO_segment_challenge,
			'P5_segment_challenge_OVO.svg'
		),
		clasificar_OVO(
			X_soybean,
			y_soybean,
			df_soybean,
			trI_soybean,
			trO_soybean,
			teI_soybean,
			teO_soybean,
			'P5_soybean_OVO.svg'
		),
		clasificar_OVO(
			X_segment_test,
			y_segment_test,
			df_segment_test,
			trI_segment_test,
			trO_segment_test,
			teI_segment_test,
			teO_segment_test,
			'P5_segment_test_OVO.svg'
		),
		clasificar_OVO(
			X_glass,
			y_glass,
			df_glass,
			trI_glass,
			trO_glass,
			teI_glass,
			teO_glass,
			'P5_glass_OVO.svg'
		),
		clasificar_OVO(
			X_ecoli,
			y_ecoli,
			df_ecoli,
			trI_ecoli,
			trO_ecoli,
			teI_ecoli,
			teO_ecoli,
			'P5_ecoli_OVO.svg'
		),
		clasificar_OVO(
			X_post_operative,
			y_post_operative,
			df_post_operative,
			trI_post_operative,
			trO_post_operative,
			teI_post_operative,
			teO_post_operative,
			'P5_post_operative_OVO.svg'
		),
		clasificar_OVO(
			X_car,
			y_car,
			df_car,
			trI_car,
			trO_car,
			teI_car,
			teO_car,
			'P5_car_OVO.svg'
		),
		clasificar_OVO(
			X_iris_2D,
			y_iris_2D,
			df_iris_2D,
			trI_iris_2D,
			trO_iris_2D,
			teI_iris_2D,
			teO_iris_2D,
			'P5_iris_2D_OVO.svg'
		)
	]
)

arrayScores_OVA = np.array(
	[
		clasificar_OVA(
			X_iris,
			y_iris,
			df_iris,
			trI_iris,
			trO_iris,
			teI_iris,
			teO_iris,
			'P5_iris_OVA.svg'
		),
		clasificar_OVO(
			X_contact_lenses,
			y_contact_lenses,
			df_contact_lenses,
			trI_contact_lenses,
			trO_contact_lenses,
			teI_contact_lenses,
			teO_contact_lenses,
			'P5_contact_lenses_OVA.svg'
		),
		clasificar_OVA(
			X_segment_challenge,
			y_segment_challenge,
			df_segment_challenge,
			trI_segment_challenge,
			trO_segment_challenge,
			teI_segment_challenge,
			teO_segment_challenge,
			'P5_segment_challenge_OVA.svg'
		),
		clasificar_OVA(
			X_soybean,
			y_soybean,
			df_soybean,
			trI_soybean,
			trO_soybean,
			teI_soybean,
			teO_soybean,
			'P5_soybean_OVA.svg'
		),
		clasificar_OVA(
			X_segment_test,
			y_segment_test,
			df_segment_test,
			trI_segment_test,
			trO_segment_test,
			teI_segment_test,
			teO_segment_test,
			'P5_segment_test_OVA.svg'
		),
		clasificar_OVA(
			X_glass,
			y_glass,
			df_glass,
			trI_glass,
			trO_glass,
			teI_glass,
			teO_glass,
			'P5_glass_OVA.svg'
		),
		clasificar_OVA(
			X_ecoli,
			y_ecoli,
			df_ecoli,
			trI_ecoli,
			trO_ecoli,
			teI_ecoli,
			teO_ecoli,
			'P5_ecoli_OVA.svg'
		),
		clasificar_OVA(
			X_post_operative,
			y_post_operative,
			df_post_operative,
			trI_post_operative,
			trO_post_operative,
			teI_post_operative,
			teO_post_operative,
			'P5_post_operative_OVA.svg'
		),
		clasificar_OVA(
			X_car,
			y_car,
			df_car,
			trI_car,
			trO_car,
			teI_car,
			teO_car,
			'P5_car_OVA.svg'
		),
		clasificar_OVA(
			X_iris_2D,
			y_iris_2D,
			df_iris_2D,
			trI_iris_2D,
			trO_iris_2D,
			teI_iris_2D,
			teO_iris_2D,
			'P5_iris_2D_OVA.svg'
		)
	]
)

arrayScores_ECOC = np.array(
	[
		clasificar_ECOC(
			X_iris,
			y_iris,
			df_iris,
			trI_iris,
			trO_iris,
			teI_iris,
			teO_iris,
			'P5_iris_ECOC.svg'
		),
		clasificar_OVO(
			X_contact_lenses,
			y_contact_lenses,
			df_contact_lenses,
			trI_contact_lenses,
			trO_contact_lenses,
			teI_contact_lenses,
			teO_contact_lenses,
			'P5_contact_lenses_ECOC.svg'
		),
		clasificar_ECOC(
			X_segment_challenge,
			y_segment_challenge,
			df_segment_challenge,
			trI_segment_challenge,
			trO_segment_challenge,
			teI_segment_challenge,
			teO_segment_challenge,
			'P5_segment_challenge_ECOC.svg'
		),
		clasificar_ECOC(
			X_soybean,
			y_soybean,
			df_soybean,
			trI_soybean,
			trO_soybean,
			teI_soybean,
			teO_soybean,
			'P5_soybean_ECOC.svg'
		),
		clasificar_ECOC(
			X_segment_test,
			y_segment_test,
			df_segment_test,
			trI_segment_test,
			trO_segment_test,
			teI_segment_test,
			teO_segment_test,
			'P5_segment_test_ECOC.svg'
		),
		clasificar_ECOC(
			X_glass,
			y_glass,
			df_glass,
			trI_glass,
			trO_glass,
			teI_glass,
			teO_glass,
			'P5_glass_ECOC.svg'
		),
		clasificar_ECOC(
			X_ecoli,
			y_ecoli,
			df_ecoli,
			trI_ecoli,
			trO_ecoli,
			teI_ecoli,
			teO_ecoli,
			'P5_ecoli_ECOC.svg'
		),
		clasificar_ECOC(
			X_post_operative,
			y_post_operative,
			df_post_operative,
			trI_post_operative,
			trO_post_operative,
			teI_post_operative,
			teO_post_operative,
			'P5_post_operative_ECOC.svg'
		),
		clasificar_ECOC(
			X_car,
			y_car,
			df_car,
			trI_car,
			trO_car,
			teI_car,
			teO_car,
			'P5_car_ECOC.svg'
		),
		clasificar_ECOC(
			X_iris_2D,
			y_iris_2D,
			df_iris_2D,
			trI_iris_2D,
			trO_iris_2D,
			teI_iris_2D,
			teO_iris_2D,
			'P5_iris_2D_ECOC.svg'
		)
	]
)

friedman = friedmanchisquare(arrayScores_dt, arrayScores_OVO, arrayScores_OVA, arrayScores_ECOC)
print("\n  FRIEDMAN => " + str(friedman) + "\n")

iman_davenport = ((10-1)*friedman[0])/(10*(4-1)-friedman[0])
print("\n  IMAN-DAVENPORT => " + str(iman_davenport) + "\n")

wilcoxon_DT_OVO = wilcoxon(arrayScores_dt, arrayScores_OVO)
print("\n  WILCOXON [DT VS. OVO] => " + str(wilcoxon_DT_OVO) + "\n")

wilcoxon_DT_OVA = wilcoxon(arrayScores_dt, arrayScores_OVA)
print("\n  WILCOXON [DT VS. OVA] => " + str(wilcoxon_DT_OVA) + "\n")

wilcoxon_DT_ECOC = wilcoxon(arrayScores_dt, arrayScores_ECOC)
print("\n  WILCOXON [DT VS. ECOC] => " + str(wilcoxon_DT_ECOC) + "\n")

wilcoxon_OVO_OVA = wilcoxon(arrayScores_OVO, arrayScores_OVA)
print("\n  WILCOXON [OVO VS. OVA] => " + str(wilcoxon_OVO_OVA) + "\n")

wilcoxon_OVO_ECOC = wilcoxon(arrayScores_OVO, arrayScores_ECOC)
print("\n  WILCOXON [OVO VS. ECOC] => " + str(wilcoxon_OVO_ECOC) + "\n")

wilcoxon_OVA_ECOC = wilcoxon(arrayScores_OVA, arrayScores_ECOC)
print("\n  WILCOXON [OVA VS. ECOC] => " + str(wilcoxon_OVA_ECOC) + "\n")