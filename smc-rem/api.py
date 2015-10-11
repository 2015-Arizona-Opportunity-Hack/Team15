from flask import Flask, render_template
from flask_restful import Api, Resource, fields, marshal_with, reqparse

from recommender import get_customers, get_recommendation


app = Flask(__name__)
api = Api(app)


@app.route('/')
def index():
    return render_template('index.html')

customer_fields = {
    'email': fields.String(attribute='Email'),
    'total_spent': fields.Integer(attribute='Total Spent'),
    'total_orders': fields.Integer(attribute='Total Orders'),
    'city': fields.String(attribute='City')
}

rec_fields = {
    'slug': fields.String(attribute='Handle'),
    'school': fields.String(attribute='Vendor'),
    'club': fields.String(attribute='Club'),
    'item': fields.String(attribute='Title'),
    'reason': fields.String(attribute='Reason'),
    'price': fields.Integer(attribute='Variant Price')
}


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


class Customers(Resource):
    @marshal_with(customer_fields)
    def get(self):
        get_customers()


class Recommendations(Resource):
    @marshal_with(rec_fields)
    def get(self):
        args = parser.parse_args()
        email = args['email']
        topN = args['topN']
        schl_w = args['school_weight']
        type_w = args['type_weight']
        text_w = args['text_weight']
        recs = get_recommendation(email=email, topN=topN, school_weight=schl_w,
                                  type_weight=type_w, text_weight=text_w)
        if recs is None:
            return []
        recs_list = [recs.iloc[idx] for idx in range(topN)]
        return recs_list


api.add_resource(Customers, '/customers')
api.add_resource(Recommendations, '/recommendations')

if __name__ == '__main__':
    app.run(debug=True)
