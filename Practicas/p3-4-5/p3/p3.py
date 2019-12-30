"""
	stats.friedmanchisquare
	stats.wilcoxon
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from pandas.api.types import is_numeric_dtype
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image

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
	train, test = dividir_ent_test(df, 0.7)
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

#Para usar un clasificador arboles de decision
def clasificar_dt(df, trainInputs, trainOutputs, testInputs, testOutputs, graphname, features, classes):
	print("\n[" + str(graphname) + "]")
	clf=DecisionTreeClassifier()
	# Train Decision Tree Classifier
	clf=clf.fit(trainInputs, trainOutputs)
	# Predict the response for test dataset
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

#Para usar un clasificador KNN
def clasificar_knn(df, trainInputs, trainOutputs, testInputs, testOutputs, graphname, classname):
	print("\n[" + str(graphname) + "]")
	clf=neighbors.KNeighborsClassifier()
	# Train KNN Classifier
	clf=clf.fit(trainInputs, trainOutputs)
	# Predict the response for test dataset
	precisionTrain = clf.score(trainInputs, trainOutputs)
	precisionTest = clf.score(testInputs, testOutputs)
	print("\tCCR train = %.2f%% | CCR test = %.2f%%" % (precisionTrain*100, precisionTest*100))
	prediccion_test = clf.predict(testInputs)
	print(prediccion_test)
	print(testOutputs)
	# graph = sns.pairplot(data=df, hue=classname)
	# graph.savefig(graphname)
	return precisionTest

#Para usar un clasificador SVM
def clasificar_svm(X, y, df, trainInputs, trainOutputs, testInputs, testOutputs, graphname, classname):
	print("\n[" + str(graphname) + "]")
	clf=svm.SVC(kernel='rbf', gamma = 'scale')
	# Train SVM Classifier
	clf=clf.fit(trainInputs, trainOutputs)
	# Predict the response for test dataset
	precisionTrain = clf.score(trainInputs, trainOutputs)
	precisionTest = clf.score(testInputs, testOutputs)
	print("\tCCR train = %.2f%% | CCR test = %.2f%%" % (precisionTrain*100, precisionTest*100))
	prediccion_test = clf.predict(testInputs)
	print(prediccion_test)
	print(testOutputs)
	# graph = sns.pairplot(data=df, hue=classname)
	# graph.savefig(graphname)
	return precisionTest

#Para usar un clasificador SVM con gridSearch
def clasificar_svm_gridSearch(X, y, df, trainInputs, trainOutputs, testInputs, testOutputs, graphname, classname):
	print("\n[" + str(graphname) + "]")
	Cs = np.logspace(-5, 15, num=11, base=2)
	Gs = np.logspace(-15, 3, num=9, base=2)
	clf=svm.SVC()
	optimo = GridSearchCV(estimator=clf, param_grid=dict(C=Cs, gamma=Gs), n_jobs=-1, cv=5)
	optimo.fit(trainInputs, trainOutputs)
	# Train SVM Classifier
	optimo.fit(trainInputs, trainOutputs)
	# Predict the response for test dataset
	precisionTrain = optimo.score(trainInputs, trainOutputs)
	precisionTest = optimo.score(testInputs, testOutputs)
	print("\tCCR train = %.2f%% | CCR test = %.2f%%" % (precisionTrain*100, precisionTest*100))
	prediccion_test = optimo.predict(testInputs)
	print(prediccion_test)
	print(testOutputs)
	return precisionTest

#Preprocesamiento de los datos de cada Dataset usado en la practica
X_iris, y_iris, df_iris, trI_iris, trO_iris, teI_iris, teO_iris = preprocesar_datos('Dataset/iris.csv')

X_contact_lenses, y_contact_lenses, df_contact_lenses, trI_contact_lenses, trO_contact_lenses, teI_contact_lenses, teO_contact_lenses = preprocesar_datos('Dataset/contact_lenses.csv')

X_ionosphere, y_ionosphere, df_ionosphere, trI_ionosphere, trO_ionosphere, teI_ionosphere, teO_ionosphere = preprocesar_datos('Dataset/ionosphere.csv')

X_soybean, y_soybean, df_soybean, trI_soybean, trO_soybean, teI_soybean, teO_soybean = preprocesar_datos('Dataset/soybean.csv')

X_diabetes, y_diabetes, df_diabetes, trI_diabetes, trO_diabetes, teI_diabetes, teO_diabetes = preprocesar_datos('Dataset/diabetes.csv')

X_glass, y_glass, df_glass, trI_glass, trO_glass, teI_glass, teO_glass = preprocesar_datos('Dataset/glass.csv')

X_labor, y_labor, df_labor, trI_labor, trO_labor, teI_labor, teO_labor = preprocesar_datos('Dataset/labor.csv')

X_vote, y_vote, df_vote, trI_vote, trO_vote, teI_vote, teO_vote = preprocesar_datos('Dataset/vote.csv')

X_car, y_car, df_car, trI_car, trO_car, teI_car, teO_car = preprocesar_datos('Dataset/car.csv')

X_bank, y_bank, df_bank, trI_bank, trO_bank, teI_bank, teO_bank = preprocesar_datos('Dataset/bank.csv')

#Generamos un array con el CCR de train y de test del clasificador de arboles de decision para cada Dataset
arrayScores_dt = np.array(
	[
		clasificar_dt(
			df_iris,
			trI_iris,
			trO_iris,
			teI_iris,
			teO_iris,
			'P3_iris_dt.svg',
			['sepallength','sepalwidth','petallength','petalwidth'],
			['Iris_setosa', 'Iris_versicolor', 'Iris_virginica']
		),
		clasificar_dt(
			df_contact_lenses,
			trI_contact_lenses,
			trO_contact_lenses,
			teI_contact_lenses,
			teO_contact_lenses,
			'P3_contact_lenses_dt.svg',
			['young','pre_presbyopic','presbyopic','hypermetrope','astigmatism','normal_tear_prod_rate'],
			['hard', 'soft', 'none']
		),
		clasificar_dt(
			df_ionosphere,
			trI_ionosphere,
			trO_ionosphere,
			teI_ionosphere,
			teO_ionosphere,
			'P3_ionosphere_dt.svg',
			['a01','a02','a03','a04','a05','a06','a07','a08','a09','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','a20','a21','a22','a23','a24','a25','a26','a27','a28','a29','a30','a31','a32','a33','a34'],
			['b', 'g']
		),
		clasificar_dt(
			df_soybean,
			trI_soybean,
			trO_soybean,
			teI_soybean,
			teO_soybean,
			'P3_soybean_dt.svg',
			['april','may','june','july','august','september','october','plant_stand_lt_normal','precip_lt_norm','precip_norm','precip_gt_norm','temp_lt_norm','temp_norm','temp_gt_norm','no_hail','crop_hist_diff_lst_year','crop_hist_same_lst_yr','crop_hist_same_lst_two_yrs','crop_hist_same_lst_sev_yrs','area_damaged_scattered','area_damaged_low_areas','area_damaged_upper_areas','area_damaged_whole_field','severity_minor','severity_pot_severe','severity_severe','seed_tmt_none','seed_tmt_fungicide','seed_tmt_other','germination_90_100','germination_80_89','germination_lt_80','plant_abnorm_growth','abnorm_leaves','leafspots_halo_absent','leafspots_halo_yellow_halos','leafspots_halo_no_yellow_halos','leafspots_marg_w_s_marg','leafspots_marg_no_w_s_marg','leafspots_marg_dna','leafspot_size_lt_1/8','leafspot_size_gt_1/8','leafspot_size_dna','leaf_shread','leaf_malf','leaf_mild_absent','leaf_mild_upper_surf','leaf_mild_lower_surf','abnorm_stem','no_lodging','stem_cankers_absent','stem_cankers_below_soil','stem_cankers_above_soil','stem_cankers_above_sec_nde','canker_lesion_dna','canker_lesion_brown','canker_lesion_dk_brown_blk','canker_lesion_tan','fruiting_bodies','external_decay_absent','external_decay_firm_and_dry','external_decay_watery','mycelium','int_discolor_none','int_discolor_brown','int_discolor_black','sclerotia','fruit_pods_norm','fruit_pods_diseased','fruit_pods_few_present','fruit_pods_dna','fruit_spots_absent','fruit_spots_colored','fruit_spots_brown_w/blk_specks','fruit_spots_distort','fruit_spots_dna','abnorm_seed','mold_growth','seed_discolor','seed_size_lt_norm','shriveling','roots_norm','roots_rotted','roots_galls_cysts'],
			['herbicide_injury', 'cyst_nematode', 'diaporthe_pod_and_stem_blight', '2_4_d_injury', 'diaporthe_stem_canker', 'charcoal_rot', 'rhizoctonia_root_rot', 'powdery_mildew', 'downy_mildew', 'bacterial_blight', 'bacterial_pustule', 'purple_seed_stain', 'phyllosticta_leaf_spot', 'brown_stem_rot', 'anthracnose', 'phytophthora_rot', 'alternarialeaf_spot', 'frog_eye_leaf_spot', 'brown_spot']
		),
		clasificar_dt(
			df_diabetes,
			trI_diabetes,
			trO_diabetes,
			teI_diabetes,
			teO_diabetes,
			'P3_diabetes_dt.svg',
			['preg','plas','pres','skin','insu','mass','pedi','age'],
			['tested_positive', 'tested_negative']
		),
		clasificar_dt(
			df_glass,
			trI_glass,
			trO_glass,
			teI_glass,
			teO_glass,
			'P3_glass_dt.svg',
			['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'],
			['vehic wind non_float', 'tableware', 'containers', 'vehic wind float', 'headlamps', 'build wind float', 'build wind non_float']
		),
		clasificar_dt(
			df_labor,
			trI_labor,
			trO_labor,
			teI_labor,
			teO_labor,
			'P3_labor_dt.svg',
			['duration','wage_increase_first_year','wage_increase_second_year','wage_increase_third_year','cost_of_living_adjustment_none','cost_of_living_adjustment_tcf','cost_of_living_adjustment_tc','working_hours','pension_none','pension_ret_allw','pension_empl_contr','standby_pay','shift_differential','no_education_allowance','statutory_holidays','vacation_below_average','vacation_average','vacation_generous','no_longterm_disability_assistance','contribution_to_dental_plan_none','contribution_to_dental_plan_half','contribution_to_dental_plan_full','no_bereavement_assistance','contribution_to_health_plan_none','contribution_to_health_plan_half','contribution_to_health_plan_full'],
			['bad', 'good']
		),
		clasificar_dt(
			df_vote,
			trI_vote,
			trO_vote,
			teI_vote,
			teO_vote,
			'P3_vote_dt.svg',
			['handicapped_infants','water_project_cost_sharing','adoption_of_the_budget_resolution','physician_fee_freeze','el_salvador_aid','religious_groups_in_schools','anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration','synfuels_corporation_cutback','education_spending','superfund_right_to_sue','crime','duty_free_exports','export_administration_act_south_africa'],
			['democrat', 'republican']
		),
		clasificar_dt(
			df_car,
			trI_car,
			trO_car,
			teI_car,
			teO_car,
			'P3_car_dt.svg',
			['buying_vhigh','buying_high','buying_med','buying_low','maint_vhigh','maint_high','maint_med','maint_low','doors_2','doors_3','doors_4','doors_more_or_equal_than_5','persons_2','persons_4','persons_more_than_4','log_boot_small','log_boot_med','log_boot_big','safety_low','safety_med','safety_high'],
			['v_good', 'good', 'acc', 'unacc']
		),
		clasificar_dt(
			df_bank,
			trI_bank,
			trO_bank,
			teI_bank,
			teO_bank,
			'P3_bank_dt.svg',
			['age','job_unemployed','job_services','job_management','job_blue_collar','job_self_employed','job_technician','job_entrepreneur','job_admin','job_student','job_housemaid','job_retired','job_unknown','marital_married','marital_single','marital_divorced','education_primary','education_secondary','education_tertiary','education_unknown','default','balance','housing','loan','contact_cellular','contact_unknown','contact_telephone','day','january','february','march','april','may','june','july','august','september','october','november','december','duration','campaign','pdays','previous','poutcome_unknown','poutcome_failure','poutcome_other','poutcome_success'],
			['yes', 'no']
		)
	]
)

#Generamos un array con el CCR de train y de test del clasificador KNN para cada Dataset
arrayScores_knn = np.array(
	[
		clasificar_knn(
			df_iris,
			trI_iris,
			trO_iris,
			teI_iris,
			teO_iris,
			'P3_iris_knn.svg',
			'class'
		),
		clasificar_knn(
			df_contact_lenses,
			trI_contact_lenses,
			trO_contact_lenses,
			teI_contact_lenses,
			teO_contact_lenses,
			'P3_contact_lenses_knn.svg',
			'contact_lenses'
		),
		clasificar_knn(
			df_ionosphere,
			trI_ionosphere,
			trO_ionosphere,
			teI_ionosphere,
			teO_ionosphere,
			'P3_ionosphere_knn.svg',
			'class'
		),
		clasificar_knn(
			df_soybean,
			trI_soybean,
			trO_soybean,
			teI_soybean,
			teO_soybean,
			'P3_soybean_knn.svg',
			'class'
		),
		clasificar_knn(
			df_diabetes,
			trI_diabetes,
			trO_diabetes,
			teI_diabetes,
			teO_diabetes,
			'P3_diabetes_knn.svg',
			'class'
		),
		clasificar_knn(
			df_glass,
			trI_glass,
			trO_glass,
			teI_glass,
			teO_glass,
			'P3_glass_knn.svg',
			'Type'
		),
		clasificar_knn(
			df_labor,
			trI_labor,
			trO_labor,
			teI_labor,
			teO_labor,
			'P3_labor_knn.svg',
			'class'
		),
		clasificar_knn(
			df_vote,
			trI_vote,
			trO_vote,
			teI_vote,
			teO_vote,
			'P3_vote_knn.svg',
			'Class'
		),
		clasificar_knn(
			df_bank,
			trI_bank,
			trO_bank,
			teI_bank,
			teO_bank,
			'P3_bank_knn.svg',
			'class'
		),
		clasificar_knn(
			df_car,
			trI_car,
			trO_car,
			teI_car,
			teO_car,
			'P3_car_knn.svg',
			'class'
		)
	]
)

#Generamos un array con el CCR de train y de test del clasificador SVM para cada Dataset
arrayScores_svm = np.array(
	[
		clasificar_svm(
			X_iris,
			y_iris,
			df_iris,
			trI_iris,
			trO_iris,
			teI_iris,
			teO_iris,
			'P3_iris_svm.svg',
			'class'
		),
		clasificar_svm(
			X_contact_lenses,
			y_contact_lenses,
			df_contact_lenses,
			trI_contact_lenses,
			trO_contact_lenses,
			teI_contact_lenses,
			teO_contact_lenses,
			'P3_contact_lenses_svm.svg',
			'contact_lenses'
		),
		clasificar_svm(
			X_ionosphere,
			y_ionosphere,
			df_ionosphere,
			trI_ionosphere,
			trO_ionosphere,
			teI_ionosphere,
			teO_ionosphere,
			'P3_ionosphere_svm.svg',
			'class'
		),
		clasificar_svm(
			X_soybean,
			y_soybean,
			df_soybean,
			trI_soybean,
			trO_soybean,
			teI_soybean,
			teO_soybean,
			'P3_soybean_svm.svg',
			'class'
		),
		clasificar_svm(
			X_diabetes,
			y_diabetes,
			df_diabetes,
			trI_diabetes,
			trO_diabetes,
			teI_diabetes,
			teO_diabetes,
			'P3_diabetes_svm.svg',
			'class'
		),
		clasificar_svm(
			X_glass,
			y_glass,
			df_glass,
			trI_glass,
			trO_glass,
			teI_glass,
			teO_glass,
			'P3_glass_svm.svg',
			'Type'
		),
		clasificar_svm(
			X_labor,
			y_labor,
			df_labor,
			trI_labor,
			trO_labor,
			teI_labor,
			teO_labor,
			'P3_labor_svm.svg',
			'class'
		),
		clasificar_svm(
			X_vote,
			y_vote,
			df_vote,
			trI_vote,
			trO_vote,
			teI_vote,
			teO_vote,
			'P3_vote_svm.svg',
			'Class'
		),
		clasificar_svm(
			X_car,
			y_car,
			df_car,
			trI_car,
			trO_car,
			teI_car,
			teO_car,
			'P3_car_svm.svg',
			'class'
		),
		clasificar_svm(
			X_bank,
			y_bank,
			df_bank,
			trI_bank,
			trO_bank,
			teI_bank,
			teO_bank,
			'P3_bank_svm.svg',
			'class'
		)
	]
)

print("\n")

#Imprimimos por pantalla los arrays obtenidos
print("\narrayScores_dt" + str(rankdata(arrayScores_dt)))
print("\narrayScores_knn" + str(rankdata(arrayScores_knn)))
print("\narrayScores_svm" + str(rankdata(arrayScores_svm)))

print("\n")

#Hacemos Wilcoxon para KNN y Arboles
wilcoxon = wilcoxon(arrayScores_knn, arrayScores_dt)

print("\n  WILCOXON => " + str(wilcoxon) + "\n")

#Hacemos Friedman para los 3 metdodos de clasificacion
friedman = friedmanchisquare(arrayScores_dt, arrayScores_knn, arrayScores_svm)

print("\n  FRIEDMAN => " + str(friedman) + "\n")

#Hacemos Iman Davenport con los datos obtenidos en Friedman
iman_davenport = ((10-1)*friedman[0])/(10*(3-1)-friedman[0])

print("\n  IMAN-DAVENPORT => " + str(iman_davenport) + "\n")

print("\n")

print("\n  GRIDSEARCH CON SVM\n")

#Hacemos el método de clustering GridSearch con SVM para cada Dataset
clasificar_svm_gridSearch(
	X_iris,
	y_iris,
	df_iris,
	trI_iris,
	trO_iris,
	teI_iris,
	teO_iris,
	'P3_iris_svm_gridSearch.svg',
	'class'
),
clasificar_svm_gridSearch(
	X_contact_lenses,
	y_contact_lenses,
	df_contact_lenses,
	trI_contact_lenses,
	trO_contact_lenses,
	teI_contact_lenses,
	teO_contact_lenses,
	'P3_contact_lenses_svm_gridSearch.svg',
	'contact_lenses'
),
clasificar_svm_gridSearch(
	X_ionosphere,
	y_ionosphere,
	df_ionosphere,
	trI_ionosphere,
	trO_ionosphere,
	teI_ionosphere,
	teO_ionosphere,
	'P3_ionosphere_svm_gridSearch.svg',
	'class'
),
clasificar_svm_gridSearch(
	X_soybean,
	y_soybean,
	df_soybean,
	trI_soybean,
	trO_soybean,
	teI_soybean,
	teO_soybean,
	'P3_soybean_svm_gridSearch.svg',
	'class'
),
clasificar_svm_gridSearch(
	X_diabetes,
	y_diabetes,
	df_diabetes,
	trI_diabetes,
	trO_diabetes,
	teI_diabetes,
	teO_diabetes,
	'P3_diabetes_svm_gridSearch.svg',
	'class'
),
clasificar_svm_gridSearch(
	X_glass,
	y_glass,
	df_glass,
	trI_glass,
	trO_glass,
	teI_glass,
	teO_glass,
	'P3_glass_svm_gridSearch.svg',
	'Type'
),
clasificar_svm_gridSearch(
	X_labor,
	y_labor,
	df_labor,
	trI_labor,
	trO_labor,
	teI_labor,
	teO_labor,
	'P3_labor_svm_gridSearch.svg',
	'class'
),
clasificar_svm_gridSearch(
	X_vote,
	y_vote,
	df_vote,
	trI_vote,
	trO_vote,
	teI_vote,
	teO_vote,
	'P3_vote_svm_gridSearch.svg',
	'Class'
),
clasificar_svm_gridSearch(
	X_car,
	y_car,
	df_car,
	trI_car,
	trO_car,
	teI_car,
	teO_car,
	'P3_car_svm_gridSearch.svg',
	'class'
),
clasificar_svm_gridSearch(
	X_bank,
	y_bank,
	df_bank,
	trI_bank,
	trO_bank,
	teI_bank,
	teO_bank,
	'P3_bank_svm_gridSearch.svg',
	'class'
)