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

        res = projector.get_player_projection(args['player_name'], int(args['n_recent_games']))
        if not res:
            return "NBA API Error."
        performance, projection = res
        return projection, 200


class ProjectionTeam(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument("abbrev")
        parser.add_argument("n_recent_games")
        args = dict(parser.parse_args())
        res = projector.get_team_projection(args['abbrev'], int(args['n_recent_games']))
        if not res:
            return 500
        performance, projection = res
        return projection, 200


class ComparePlayerProjection(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument("player_name_A")
        parser.add_argument("player_name_B")
        parser.add_argument("n_recent_games")
        args = dict(parser.parse_args())

        res = projector.compare_players(args['player_name_A'], args['player_name_B'], int(args['n_recent_games']))
        if not res:
            return 500
        return res, 200


PORT = 4444
api.add_resource(ProjectionPlayer, '/api/project/player')
api.add_resource(ProjectionTeam, '/api/project/team')
api.add_resource(ComparePlayerProjection, '/api/compare/player')
app.run(port=PORT, debug=True)

