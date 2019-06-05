
import unittest

from sklearn.isotonic import *
from sklearn.linear_model import *
from sklearn.discriminant_analysis import *
from sklearn.mixture import *
from sklearn.kernel_ridge import *
from sklearn.gaussian_process import *
from sklearn.decomposition import *
from sklearn.feature_selection import *
from sklearn.cluster import *
from sklearn.covariance import *
from sklearn.manifold import *
from sklearn.semi_supervised import *
from sklearn.calibration import *
from sklearn.neural_network import *
from sklearn.cross_decomposition import *
from sklearn.multiclass import *
from sklearn.multioutput import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from pypadre.core.visitors import DictVisitor, ListVisitor
from pypadre.core.visitors.scikit import SciKitVisitor

import pprint


class TestExperimentVisitor(unittest.TestCase):
    def test_extract_string(self):
        class TestClass(object):
            def __init__(self):
                self.test = "attribute"

        v = DictVisitor({"test": "found"})
        d = v.extract(TestClass(), {})

        print(d)

        self.assertIn("found", d)
        self.assertEqual(d["found"], "attribute")

    def test_extract_dict(self):
        class TestClass(object):
            def __init__(self):
                self.test = {"rec": "attribute"}

        v = DictVisitor({"test": {"rec" : "found"}})
        d = v.extract(TestClass(), {})

        self.assertIn("found", d)
        self.assertEqual(d["found"], "attribute")

    def test_extract_list(self):
        class TestClass(object):
            def __init__(self):
                self.test = ["a", "b", "c"]

        v = DictVisitor({"test": ListVisitor("items", "value")})
        d = v.extract(TestClass(), {})

        print(d)



class TestSciKitExperimentVisitor(unittest.TestCase):

    def test_extract_pipeline(self):
        estimators = [('step0', PCA()), ('step1', Ridge())]
        pipe = Pipeline(estimators)

        d = SciKitVisitor(pipe)

        self.assertIn("steps", d[0])
        for s in d[0]["steps"]:
            for k,v in s.items():
                print(k,v)

        #pprint.PrettyPrinter().pprint(d)

    def extract_type_object(self, type):

        print("Testing " + str(type))

        lreg = type()

        d = SciKitVisitor(lreg)

        self.assertIn("steps", d[0])
        #for s in d[0]["steps"]:
        #    for k,v in s.items():
        #        print(k,v)

    def test_extract_linear_regression(self):
        self.extract_type_object(LinearRegression)

    def test_extract_ridge_regression(self):
        self.extract_type_object(Ridge)

    def test_extract_lasso(self):
        self.extract_type_object(Lasso)

    def test_extract_multi_task_lasso(self):
        self.extract_type_object(MultiTaskLasso)

    def test_extract_elastic_net(self):
        self.extract_type_object(ElasticNet)

    def test_extract_multi_task_elastic_net(self):
        self.extract_type_object(MultiTaskElasticNet)

    def test_extract_least_angle_regression(self):
        self.extract_type_object(Lars)

    def test_extract_lasso_least_angle_regression(self):
        self.extract_type_object(LassoLars)

    def test_extract_orthogonal_matching_pursuit_model(self):
        self.extract_type_object(OrthogonalMatchingPursuit)

#    def test_extract_n_target_orthogonal_matching_pursuit(self):
#        self.extract_type_object(orthogonal_mp)

    def test_extract_bayesian_ridge_regression(self):
        self.extract_type_object(BayesianRidge)

    def test_extract_automatic_relevance_determination_regression(self):
        self.extract_type_object(ARDRegression)

    def test_extract_logistic_regression(self):
        self.extract_type_object(LogisticRegression)

    def test_extract_stochastic_gradient_descent_classifier(self):
        self.extract_type_object(SGDClassifier)

    def test_extract_stochastic_gradient_descent_regressor(self):
        self.extract_type_object(SGDRegressor)

    def test_extract_perceptron(self):
        self.extract_type_object(Perceptron)

    def test_extract_passive_aggressive_classifier(self):
        self.extract_type_object(PassiveAggressiveClassifier)

    def test_extract_passive_aggressive_regressor(self):
        self.extract_type_object(PassiveAggressiveRegressor)

    def test_extract_random_sample_consensus_regressor(self):
        self.extract_type_object(RANSACRegressor)

    def test_extract_theil_sen_estimator(self):
        self.extract_type_object(TheilSenRegressor)

    def test_extract_huber_regressor(self):
        self.extract_type_object(HuberRegressor)

    def test_extract_linear_discriminant_analysis(self):
        self.extract_type_object(LinearDiscriminantAnalysis)

    def test_extract_quadratic_discriminant_analysis(self):
        self.extract_type_object(QuadraticDiscriminantAnalysis)

    def test_extract_kernel_ridge_regression(self):
        self.extract_type_object(KernelRidge)

    def test_extract_c_support_vector_classification(self):
        self.extract_type_object(SVC)

    def test_extract_nu_support_vector_classification(self):
        self.extract_type_object(NuSVC)

    def test_extract_linear_support_vector_classification(self):
        self.extract_type_object(LinearSVC)

    def test_extract_epsilon_support_vector_regression(self):
        self.extract_type_object(SVR)

    def test_extract_nu_support_vector_regression(self):
        self.extract_type_object(NuSVR)

    def test_extract_linear_support_vector_regression(self):
        self.extract_type_object(LinearSVR)

    def test_extract_one_class_support_vector_machine(self):
        self.extract_type_object(OneClassSVM)

    def test_extract_nearest_neighbours(self):
        self.extract_type_object(NearestNeighbors)

#    def test_extract_k_d_tree(self):
#        self.extract_type_object(KDTree)

#   def test_extract_ball_tree(self):
#       self.extract_type_object(BallTree)

    def test_extract_radius_neighbors_classifier(self):
        self.extract_type_object(RadiusNeighborsClassifier)

    def test_extract_k_nn_classifier(self):
        self.extract_type_object(KNeighborsClassifier)

    def test_extract_k_nn_regressor(self):
        self.extract_type_object(KNeighborsRegressor)

    def test_extract_radius_neighbors_regressor(self):
        self.extract_type_object(RadiusNeighborsRegressor)

    def test_extract_nearest_centroid_classifier(self):
        self.extract_type_object(NearestCentroid)

    def test_extract_gaussian_process_regression(self):
        self.extract_type_object(GaussianProcessRegressor)

    def test_extract_gaussian_process_classification(self):
        self.extract_type_object(GaussianProcessClassifier)

    def test_extract_partial_least_squares_regression(self):
        self.extract_type_object(PLSRegression)

    def test_extract_2_blocks_canonical_partial_least_squares(self):
        self.extract_type_object(PLSCanonical)

    def test_extract_partial_least_squares_singular_value_decomposition(self):
        self.extract_type_object(PLSSVD)

    def test_extract_canonical_correlation_analysis(self):
        self.extract_type_object(CCA)

    def test_extract_gaussian_naive_bayes(self):
        self.extract_type_object(GaussianNB)

    def test_extract_multinomial_naive_bayes_classifier(self):
        self.extract_type_object(MultinomialNB)

    def test_extract_bernoulli_naive_bayes_classifier(self):
        self.extract_type_object(BernoulliNB)

    def test_extract_decision_tree_classifier(self):
        self.extract_type_object(DecisionTreeClassifier)

    def test_extract_decision_tree_regressor(self):
        self.extract_type_object(DecisionTreeRegressor)

    def test_extract_extra_trees_regressor(self):
        self.extract_type_object(ExtraTreesRegressor)

    def test_extract_extra_trees_classifier(self):
        self.extract_type_object(ExtraTreesClassifier)

    def test_extract_random_forest_regressor(self):
        self.extract_type_object(RandomForestRegressor)

    def test_extract_random_forest_classifier(self):
        self.extract_type_object(RandomForestClassifier)

    def test_extract_bagging_classifier(self):
        self.extract_type_object(BaggingClassifier)

    def test_extract_bagging_regressor(self):
        self.extract_type_object(BaggingRegressor)

#    def test_extract_vating_classifier(self):
#        self.extract_type_object(VotingClassifier)

    def test_extract_gradient_boosting_for_regression(self):
        self.extract_type_object(GradientBoostingRegressor)

    def test_extract_gradient_boosting_for_classification(self):
        self.extract_type_object(GradientBoostingClassifier)

    def test_extract_AdaBoost_regressor(self):
        self.extract_type_object(AdaBoostRegressor)

    def test_extract_AdaBoost_classifier(self):
        self.extract_type_object(AdaBoostClassifier)

    def test_extract_random_trees_ensemble(self):
        self.extract_type_object(RandomTreesEmbedding)

#    def test_extract_classifier_chain(self):
#        self.extract_type_object(ClassifierChain)

#    def test_extract_multi_target_regression(self):
#        self.extract_type_object(MultiOutputRegressor)

#    def test_extract_output_code_classifier(self):
#        self.extract_type_object(OutputCodeClassifier)

#    def test_extract_one_vs_one_classifier(self):
#        self.extract_type_object(OneVsOneClassifier)

#    def test_extract_one_vs_rest_classifier(self):
#        self.extract_type_object(OneVsRestClassifier)

#    def test_extract_select_from_model(self):
#        self.extract_type_object(SelectFromModel)

#    def test_extract_recursive_feature_elimination_with_cross_validation(self):
#        self.extract_type_object(RFECV)

#    def test_extract_recursive_feature_elimination(self):
#        self.extract_type_object(RFE)

    def test_extract_univariate_feature_selector(self):
        self.extract_type_object(GenericUnivariateSelect)

    def test_extract_select_familiy_wise_error_rate(self):
        self.extract_type_object(SelectFwe)

    def test_extract_select_estimated_false_discovery_rate(self):
        self.extract_type_object(SelectFdr)

    def test_extract_select_below_false_positive_rate(self):
        self.extract_type_object(SelectFpr)

    def test_extract_select_percentile(self):
        self.extract_type_object(SelectPercentile)

    def test_extract_select_k_best(self):
        self.extract_type_object(SelectKBest)

    def test_extract_variance_threshold(self):
        self.extract_type_object(VarianceThreshold)

    def test_extract_label_probagation(self):
        self.extract_type_object(LabelPropagation)

    def test_extract_label_spreading(self):
        self.extract_type_object(LabelSpreading)

    def test_extract_isotonic_regression(self):
        self.extract_type_object(IsotonicRegression)

    def test_extract_probability_calibration_with_cross_validation(self):
        self.extract_type_object(CalibratedClassifierCV)

    def test_extract_multi_layer_perceptron_regressor(self):
        self.extract_type_object(MLPRegressor)

    def test_extract_multi_layer_perceptron_classifier(self):
        self.extract_type_object(MLPClassifier)

    def test_extract_Gaussian_mixture(self):
        self.extract_type_object(GaussianMixture)

    def test_extract_Bayesian_Gaussian_mixture(self):
        self.extract_type_object(BayesianGaussianMixture)

    def test_extract_isomap_embedding(self):
        self.extract_type_object(Isomap)

#    def test_extract_locally_linear_embedding_analysis(self):
#        self.extract_type_object(locally_linear_embedding)

    def test_extract_object_oriented_locally_linear_embedding_analysis(self):
        self.extract_type_object(LocallyLinearEmbedding)

    def test_extract_multidimensional_scaling(self):
        self.extract_type_object(MDS)

#    def test_extract_spectral_embedding(self):
#        self.extract_type_object(spectral_embedding)

    def test_extract_object_oriented_spectral_embedding(self):
        self.extract_type_object(SpectralEmbedding)

    def test_extract_t_distributed_stochastic_neighbor_embedding(self):
        self.extract_type_object(TSNE)

    def test_extract_k_means_clustering(self):
        self.extract_type_object(KMeans)

    def test_extract_mini_batch_k_means_clustering(self):
        self.extract_type_object(MiniBatchKMeans)

    def test_extract_affinity_propagation_clustering(self):
        self.extract_type_object(AffinityPropagation)

    def test_extract_mean_shift_clustering_with_flat_kernel(self):
        self.extract_type_object(MeanShift)

    def test_extract_spectral_clustering(self):
        self.extract_type_object(SpectralClustering)

    def test_extract_agglomerative_clustering(self):
        self.extract_type_object(AgglomerativeClustering)

    def test_extract_feature_agglomerative_clustering(self):
        self.extract_type_object(FeatureAgglomeration)

    def test_extract_DBSCAN(self):
        self.extract_type_object(DBSCAN)

    def test_extract_birch_clustering(self):
        self.extract_type_object(Birch)

    def test_extract_spectral_co_clustering(self):
        self.extract_type_object(SpectralCoclustering)

    def test_extract_spectral_biclustering(self):
        self.extract_type_object(SpectralBiclustering)

    def test_extract_principal_component_analysis(self):
        self.extract_type_object(PCA)

    def test_extract_incremental_principal_component_analysis(self):
        self.extract_type_object(IncrementalPCA)

    def test_extract_kernel_principal_component_analysis(self):
        self.extract_type_object(KernelPCA)

    def test_extract_sparse_principal_component_analysis(self):
        self.extract_type_object(SparsePCA)

    def test_extract_mini_batch_sparse_principal_component_analysis(self):
        self.extract_type_object(MiniBatchSparsePCA)

    def test_extract_truncated_single_value_decomposition(self):
        self.extract_type_object(TruncatedSVD)

#    def test_extract_sparse_coding(self):
#        self.extract_type_object(SparseCoder)

    def test_extract_dictionary_learning(self):
        self.extract_type_object(DictionaryLearning)

    def test_extract_mini_batch_dictionary_learning(self):
        self.extract_type_object(MiniBatchDictionaryLearning)

    def test_extract_factor_analysis(self):
        self.extract_type_object(FactorAnalysis)

    def test_extract_fast_independent_component_analysis(self):
        self.extract_type_object(FastICA)

    def test_extract_non_negative_matrix_factorization(self):
        self.extract_type_object(NMF)

    def test_extract_latent_Dirichlet_allocation(self):
        self.extract_type_object(LatentDirichletAllocation)

    def test_extract_elliptice_envelope(self):
        self.extract_type_object(EllipticEnvelope)

    def test_extract_isolation_forest(self):
        self.extract_type_object(IsolationForest)

    def test_extract_local_outlier_factor(self):
        self.extract_type_object(LocalOutlierFactor)

    def test_extract_kernel_density(self):
        self.extract_type_object(KernelDensity)

    def test_extract_Bernoulli_restricted_Boltzmann_machines(self):
        self.extract_type_object(BernoulliRBM)




if __name__ == '__main__':
    unittest.main()
