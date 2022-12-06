from flask import Flask
from flask_restful import Api, Resource, reqparse
import pandas as pd

from backend.ProjectionModels import ProjectionModels

app = Flask(__name__)
api = Api(app)

nba = pd.read_csv("data/nba.csv", low_memory=False)
projector = ProjectionModels(nba)
n_recent_games = 50


class ProjectionPlayer(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument("player_name")
        parser.add_argument("n_recent_games")
        args = dict(parser.parse_args())

        performance, projection = projector.get_player_projection(args['player_name'], int(args['n_recent_games']))
        return projection, 200


class ProjectionTeam(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument("abbrev")
        parser.add_argument("n_recent_games")
        args = dict(parser.parse_args())
        performance, projection = projector.get_team_projection(args['abbrev'], int(args['n_recent_games']))
        return projection, 200


class CompareProjection(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument("player_name_A")
        parser.add_argument("player_name_B")
        parser.add_argument("n_recent_games")
        args = dict(parser.parse_args())

        A, B = projector.compare_players(args['player_name_A'], args['player_name_B'], int(args['n_recent_games']))
        comparison = {
            'A': {
                'player': args['player_name_A'],
                'performance': A[0],
                'projection': A[1],
            },
            'B': {
                'player': args['player_name_B'],
                'performance': B[0],
                'projection': B[1]
            }
        }
        return comparison, 200


PORT = 4444
api.add_resource(ProjectionPlayer, '/api/player')
api.add_resource(ProjectionTeam, '/api/team')
api.add_resource(CompareProjection, '/api/compare')
app.run(port=PORT, debug=True)
