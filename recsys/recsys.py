from surprise import SVD, SVDpp, NormalPredictor, KNNBaseline, KNNBasic
from surprise.model_selection import train_test_split, LeaveOneOut, GridSearchCV
from surprise import Dataset, Reader
from surprise import accuracy
from collections import defaultdict
import numpy as np
import pandas as pd
import csv, sys
import itertools
import pickle

# Path to datasets
ratings_file_path = './datasets/final_ratings.csv'
products_file_path = './datasets/final_products.csv'


class DatasetHelper:
    # product id to name mapper and vice versa
    productIdToName = {}
    productNameToId = {}
    productPopularityRankings = defaultdict(int)

    def __init__(self, verbose=True):
        self.verbose = verbose

    def load(self):
        print('Loading Dataset ..') if self.verbose else None
        self.ratingsDataset = Dataset.load_from_file(ratings_file_path, reader=Reader(
            line_format='user item rating', skip_lines=1, sep=','))
        # Create a product name resolver for later use
        with open(products_file_path, newline='\n') as productsFile:
            productFileReader = csv.reader(productsFile)
            next(productFileReader)  # skip header
            for row in productFileReader:
                self.productIdToName[row[0]] = row[1]
                self.productNameToId[row[1]] = row[0]

    def getProductName(self, id):
        return self.productIdToName[id]

    def getProductId(self, name):
        return self.productNameToId[name]

    def getRatingsDataset(self):
        return self.ratingsDataset

    def getPopularityRankings(self):
        if not hasattr(self, 'productPopularityRankings'):
            self.findPopularityRankings()
        return self.productPopularityRankings

    def getFullTrainSet(self):
        if not hasattr(self, 'fullTrainSet'):
            self.buildFullTrainTestSet()
        return self.fullTrainSet

    def getFullAntiTestSet(self):
        if not hasattr(self, 'fullAntiTestSet'):
            self.buildFullTrainTestSet()
        return self.fullAntiTestSet

    def getSplitTrainSet(self):
        if not hasattr(self, 'splitTrainSet'):
            self.builSplitTrainTestSet()
        return self.splitTrainSet

    def getSplitTestSet(self):
        if not hasattr(self, 'splitTestSet'):
            self.builSplitTrainTestSet()
        return self.splitTestSet

    def getLOOCVTrain(self):
        if not hasattr(self, 'LOOCVTrain'):
            self.buildLeaveOneOutTrainTestSplit()
        return self.LOOCVTrain

    def getLOOCVTest(self):
        if not hasattr(self, 'LOOCVTest'):
            self.buildLeaveOneOutTrainTestSplit()
        return self.LOOCVTest

    def getLOOCVAntiTestSet(self):
        if not hasattr(self, 'LOOCVAntiTestSet'):
            self.buildLeaveOneOutTrainTestSplit()
        return self.LOOCVAntiTestSet

    def getSimilarities(self):
        return self.similarities

    def findPopularityRankings(self):
        print('Calculating popularity Rankings ..') if self.verbose else None
        productOccurence = defaultdict(int)
        with open(ratings_file_path, newline='\n') as ratingsFile:
            ratingFileReader = csv.reader(ratingsFile)
            next(ratingFileReader)
            for row in ratingFileReader:
                productOccurence[row[1]] += 1
        rankCount = 1
        for productId, productOccurence in sorted(productOccurence.items(), key=lambda x: x[1], reverse=True):
            self.productPopularityRankings[productId] = rankCount
            rankCount += 1

    def buildFullTrainTestSet(self):
        print('Building full training set ..') if self.verbose else None
        # for evaluating overall properties
        self.fullTrainSet = self.ratingsDataset.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()

    def builSplitTrainTestSet(self, test_size=0.3, random_state=1):
        print('Building split train test ..') if self.verbose else None
        # for measuring accuracy
        self.splitTrainSet, self.splitTestSet = train_test_split(
            self.ratingsDataset, test_size=test_size, random_state=random_state)

    def buildLeaveOneOutTrainTestSplit(self, n_splits=1, random_state=1):
        print('Building LeaveOneOut train test split ..') if self.verbose else None
        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        for train, test in LeaveOneOut(n_splits=n_splits, random_state=random_state).split(self.ratingsDataset):
            self.LOOCVTrain = train
            self.LOOCVTest = test
        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()

    def computeSimilarityMatrix(self):
        print('Computing smimilarity matrix ..') if self.verbose else None
        # Compute similarty matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.similarities = KNNBaseline(
            sim_options=sim_options).fit(self.fullTrainSet)

    def GetAntiTestSetForUser(self, testSubject):
        trainset = self.getFullTrainSet()
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(testSubject))
        user_items = set([j for (j, _) in trainset.ur[u]]) #get ratings for the user
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset


class EvaluationHelper:
    algorithms = {}

    def __init__(self, datasetHelper, verbose=True):
        self.verbose = verbose
        self.datasetHelper = datasetHelper

    def addAlgorithm(self, algorithmName, algorithm):
        print('Algorithm ' + algorithmName +
              ' added!') if self.verbose else None
        # we can add different algorithms so we can compare their results
        self.algorithms[algorithmName] = algorithm

    def evaluate(self, doTopN=False, n=10, minimumRating=4.0):
        self.n = n
        self.minimumRating = minimumRating
        print('Starting evaluations ..') if self.verbose else None
        allEvaluations = {}
        for name, algorithm in self.algorithms.items():
            metrics = {}
            metrics['RMSE'], metrics['MAE'] = self.computeAccuracy(algorithm)
            if doTopN:
                metrics['HR'], metrics['cHR'], metrics['ARHR'], metrics['coverage'] = self.evaluateTopNWithLOOCV(
                    algorithm)
            allEvaluations[name] = metrics
        # print the metrics now
        if not doTopN:
            print("{:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE"))
            for name, metrics in allEvaluations.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"]))
        else:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE", "HitRate", 'cHitRate', 'ARHR', 'Coverage'))
            for name, metrics in allEvaluations.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"], metrics['HR'], metrics['cHR'], metrics['ARHR'], metrics['coverage']))

    def evaluateTopNWithLOOCV(self, algorithm):
        print('Evaluating top 10 recommender with LOOCV:') if self.verbose else None
        algorithm.fit(self.datasetHelper.getLOOCVTrain())
        leftOutPredictions = algorithm.test(self.datasetHelper.getLOOCVTest())
        # Build predictions for all ratings not in the training set
        predictionsNotInTrainingSet = algorithm.test(
            self.datasetHelper.getLOOCVAntiTestSet())
        # compute top 10 recommendations for each user not in training set
        topNPredictedForEachUser = self.getTopN(predictionsNotInTrainingSet)
        hitRate = self.computeHitRate(
            topNPredictedForEachUser, leftOutPredictions)
        commulativeHitRate = self.computeComulativeHitRate(
            topNPredictedForEachUser, leftOutPredictions)
        averageReciporcalHitRate = self.computeAverageReciprocalHitRate(
            topNPredictedForEachUser, leftOutPredictions)
        leftOutPredictions, predictionsNotInTrainingSet, topNPredictedForEachUser = None, None, None  # Free up memory
        # Lets evaluate properties of recommendation on full training set
        algorithm.fit(self.datasetHelper.getFullTrainSet())
        allPredictions = algorithm.test(
            self.datasetHelper.getFullAntiTestSet())
        topNPredictedForEachUser = self.getTopN(allPredictions)
        userCoverage = self.computeUserCoverage(
            topNPredictedForEachUser, self.datasetHelper.getFullTrainSet().n_users)
        return hitRate, commulativeHitRate, averageReciporcalHitRate, userCoverage

    def getTopN(self, predictions):
        print('Computing Top N ...') if self.verbose else None
        topN = defaultdict(list)
        for userId, productId, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating > self.minimumRating):
                topN[int(userId)].append(((int(productId)), estimatedRating))
        for userId, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userId)] = ratings[:self.n]
        return topN

    def computeAccuracy(self, algorithm):
        print('Computing accuracy RMSE & MAE ..') if self.verbose else None
        predictions = algorithm.fit(self.datasetHelper.getSplitTrainSet()).test(
            self.datasetHelper.getSplitTestSet())
        return accuracy.rmse(predictions, verbose=False), accuracy.mae(predictions, verbose=False)

    def computeHitRate(self, predicted, leftoutPredictions):
        print('Computing Hit Rate ..') if self.verbose else None
        # See how often we recommended a product the user actually rated
        hits, total = 0, 0
        # For each left-out rating
        for leftOut in leftoutPredictions:
            userId, leftOutProduct = leftOut[0], leftOut[1]
            hit = False
            for productId, predictedRating in predicted[int(userId)]:
                if int(leftOutProduct) == int(productId):
                    hit = True
                    break
            if hit:
                hits += 1
            total += 1
        # compute overall precision
        return hits/total

    def computeComulativeHitRate(self, topNPredictedForEachUser, leftOutPredictions, ratingCutoff=4.0):
        print('Computing Commulative Hit Rate ..') if self.verbose else None
        hits, total = 0, 0
        for userId, leftOutproductId, actualRating, predictedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for productId, predictedRating in topNPredictedForEachUser[int(userId)]:
                    if int(leftOutproductId) == productId:
                        hit = True
                        break
                if hit:
                    hits += 1
                total += 1
        # compute overall precision
        return hits/total

    def computeAverageReciprocalHitRate(self, topNPredictedForEachUser, leftOutPredictions):
        print('Computing Average Reciprocal Hit Rate .. ') if self.verbose else None
        total, summation = 0, 0
        for userId, leftOutProductId, actualRating, predictedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for productId, predictedRating in topNPredictedForEachUser[int(userId)]:
                rank += 1
                if int(leftOutProductId) == productId:
                    hitRank = rank
                    break
            if hitRank > 0:
                summation += 1.0/hitRank
            total += 1
        return summation/total

    def computeUserCoverage(self, topNPredictedForEachUser, numberOfUsers, ratingThreshold=4.0):
        print('Computing user coverage ..') if self.verbose else None
        # user coverage with a minimum predicted rating of 4.0:
        # What percentage of users have at least one "good" recommendation
        hits = 0
        for userId in topNPredictedForEachUser.keys():
            hit = False
            for productId, predictedRating in topNPredictedForEachUser[int(userId)]:
                if predictedRating >= ratingThreshold:
                    hit = True
                    break
            if hit:
                hits += 1
        return hits/numberOfUsers

    def getTopNRecommendationForUser(self, subjectUser="5ed66660bee1097ce59560ec", n=10, saveModel=False):
        for name, algo in self.algorithms.items():
            print('RECOMMENDATIONS FOR USER: ', subjectUser, 'ALGORITHM :', name)
            # get full trains set
            algo.fit(self.datasetHelper.getFullTrainSet())
            if saveModel:
                pickle.dump(algo, open('./models/'+name+'_model.sav', 'wb'))
            # get recommendations using anti-train set
            predictions = algo.test(self.datasetHelper.GetAntiTestSetForUser(subjectUser))
            final_recommendations = []
            for userId, productId, actualRating, estimatedRating, _ in predictions:
                final_recommendations.append((productId, estimatedRating))
            final_recommendations.sort(key=lambda x: x[1], reverse=True)
            for rating in final_recommendations[:n]:
                print('-----------------------------------')
                print('ProductName: ', self.datasetHelper.getProductName(str(rating[0])))
                print('Score: ', rating[1])
                print('-----------------------------------')


def getBestParameters(algorithm, dataset,verbose=True):
    print("Searching for best parameters...") if verbose else None
    param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
              'n_factors': [50, 100]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(dataset)
    print('Best RMSE and MAE found after adjusting hyper parameters', gs.best_score['rmse'], gs.best_score['mae']) if verbose else None
    print('Best parameters found ', gs.best_params) if verbose else None
    return gs.best_params['rmse']

if __name__ == '__main__':
    datasetHelper = DatasetHelper(verbose=True)
    datasetHelper.load()
    evaluationHelper = EvaluationHelper(datasetHelper,verbose=True)
    evaluationHelper.addAlgorithm('SVD', SVD())
    SVDTuned = SVD(n_epochs=20, lr_all = 0.005, n_factors = 50)
    evaluationHelper.addAlgorithm('SVD Tuned', SVDTuned)
    print('Evaluating Untuned-SVD vs Tuned SVD')
    Start evaluation
    evaluationHelper.evaluate(doTopN=True)
    evaluationHelper.getTopNRecommendationForUser(saveModel=True)
