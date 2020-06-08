from flask import Flask, request, url_for, jsonify
from surprise import Dataset, Reader
import csv
import sys
import pickle

# my saved model path
model_path = './app/models/SVDTuned_model.sav'
ratings_file_path = './app/data/final_ratings.csv'

# get anti-test set for user


def GetAntiTestSetForUser(testSubject, fullTrainingSet):
    trainset = fullTrainingSet
    fill = trainset.global_mean
    anti_testset = []
    u = trainset.to_inner_uid(testSubject)
    user_items = set([j for (j, _) in trainset.ur[u]]
                     )  # get ratings for the user
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                     i in trainset.all_items() if
                     i not in user_items]
    return anti_testset


# Load Dataset
ratingsDataset = Dataset.load_from_file(ratings_file_path, reader=Reader(
    line_format='user item rating', skip_lines=1, sep=','))

# build a full training set
fullTrainingSet = ratingsDataset.build_full_trainset()

# Load trained model
model = pickle.load(open(model_path, 'rb'))

# app instance
app = Flask(__name__)


@app.route("/api/getRecs", methods=["POST"])
def getRecs():
    try:
        #force=True, above, is necessary if MIME type not 'application/json'
        input_json = request.get_json(force=True)
        predictions = model.test(GetAntiTestSetForUser(
            input_json['userId'], fullTrainingSet))
        limit = 50
        if 'limit' in input_json:
            limit = input_json['limit']
        final_recommendations = []
        for userId, productId, actualRating, estimatedRating, _ in predictions:
            final_recommendations.append((productId, estimatedRating))
            final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return jsonify(final_recommendations[:limit])
    except Exception as err:
        return jsonify({'err': 'Some error occured while getting recommendations'})

# if __name__ == "__main__":
#     app.run(debug=True)
