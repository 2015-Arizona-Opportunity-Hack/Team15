from flask import Flask
from flask_restful import Api, Resource, fields, marshal_with, reqparse

from recommender import get_recommendation


app = Flask(__name__)
api = Api(app)

rec_fields = {
    'slug': fields.String(attribute='Handle'),
    'school': fields.String(attribute='Vendor'),
    'club': fields.String(attribute='Club'),
    'item': fields.String(attribute='Title'),
    'reason': fields.String(attribute='Reason'),
    'price': fields.Integer(attribute='Variant Price')
}

recs_fields = {
    'recs': fields.List(fields.Nested(rec_fields))
}


# class RecommendationDao(object):
#     def __init__(self, slug, school, club, item, reason, price):
#         self.slug = slug
#         self.school = school
#         self.club = club
#         self.item = item
#         self.reason = reason
#         self.price = price


parser = reqparse.RequestParser()
parser.add_argument('email', help='Customer to recommend to')
parser.add_argument('topN', default=5, type=int,
                    help='Num of recommendations to return')
parser.add_argument('school_weight', default=1, type=float,
                    help='Weight to apply to school matching')
parser.add_argument('type_weight', default=1, type=float,
                    help='Weight to apply to type matching')
parser.add_argument('text_weight', default=1, type=float,
                    help='Weight to apply to text matching')


class Recommendations(Resource):
    @marshal_with(recs_fields)
    def get(self):
        args = parser.parse_args()
        email = args['email']
        topN = args['topN']
        schl_w = args['school_weight']
        type_w = args['type_weight']
        text_w = args['text_weight']
        recs = get_recommendation(email=email, topN=topN, school_weight=schl_w,
                                  type_weight=type_w, text_weight=text_w)
        recs_list = [recs.iloc[idx] for idx in range(topN)]
        return {'recs': recs_list}

api.add_resource(Recommendations, '/recommendation')

if __name__ == '__main__':
    app.run(debug=True)
