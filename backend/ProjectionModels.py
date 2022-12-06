import os
import time

import joblib as joblib
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class ProjectionModels:
    def __init__(self, data):
        self.raw_data = data
        # Default columns used for model training
        self.t_features = ['efg_pct', 'tov', 'drb', 'orb', 'ft_pct', 'pts', 'ast',
                           'minutes', 'ts_pct', 'stl', 'blk', 'fga']
        # Default columns to be predicted by models
        self.p_features = ['pts', 'ast', 'drb', 'orb', 'stl', 'blk', 'tov', 'fga', 'minutes']

        self.t_features_data = self.raw_data[self.t_features]

        self.le = LabelEncoder()
        self.oh = OneHotEncoder(sparse=False)

        if len(os.listdir('models')) == 0:
            self.models = self.train_models()
            self.serialize()
        else:
            self.models = {}
            for feature in self.p_features:
                self.models[feature] = joblib.load(f"models/{feature}.mdl")

    def serialize(self):
        for mdl in self.models:
            joblib.dump(self.models[mdl], f"models/{mdl}.mdl")

    def retrain_models(self):
        dir = 'models'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        self.models = self.train_models()
        self.serialize()

    def train_models(self):
        models = {}
        for feature in self.p_features:
            print(f"\n================ {feature}")
            start = time.time()

            X, y = self.t_features_data.drop(feature, axis=1), self.t_features_data[[feature]]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            r_score = {}
            for k in range(1, 10):
                k_value = k + 1

                knn = KNeighborsRegressor(n_neighbors=k_value)
                knn.fit(X, y)
                y_pred = knn.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                if r2 != 1:
                    r_score[k_value] = r2_score(y_test, y_pred)
            k = max(r_score, key=lambda x: r_score[x])

            end = time.time()
            print(f"\nTrained {feature} model in {end - start}s")

            print(f"\nk value for max R2: {k}")
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X, y)
            models[feature] = model

            y_pred = model.predict(X_test)
            print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
            print(f"R2: {metrics.r2_score(y_test, y_pred)}")
        return models

    def get_player_projection(self, player, n_recent_games=5):
        raw_player_data = self.raw_data[self.raw_data.player == player]
        copy_player_data = raw_player_data.copy()

        recent_games = []
        for i in range(n_recent_games):
            most_recent_game_date = copy_player_data['game_date'].max()  # gets players most recent game
            if most_recent_game_date != most_recent_game_date:  # nan check
                break
            recent_games.append(most_recent_game_date)
            copy_player_data = copy_player_data[copy_player_data.game_date != most_recent_game_date]

        recent_player_stats = raw_player_data[raw_player_data.game_date.isin(recent_games)][self.t_features]
        mean_player_stats = recent_player_stats.mean().to_frame().transpose()

        projections = {}
        for model in self.models:
            knn = self.models[model]
            projections[model] = knn.predict(mean_player_stats.drop(model, axis=1))[0][0]
        return mean_player_stats[self.p_features].iloc[0].to_dict(), projections

    def get_team_projection(self, abbrev, n_recent_games=5):
        team_data = self.raw_data[self.raw_data.Team_Abbrev == abbrev]
        most_recent_date = team_data['game_date'].max()
        most_recent_team = set(team_data[team_data.game_date == most_recent_date]['player'].tolist())

        team_projections = {}
        team_performance = {}
        for player in most_recent_team:
            performance, projections = self.get_player_projection(player, n_recent_games)

            for stat in projections:
                try:
                    team_projections[stat] += projections[stat]
                except KeyError:
                    team_projections[stat] = projections[stat]

                try:
                    team_performance[stat] += performance[stat]
                except KeyError:
                    team_performance[stat] = performance[stat]

        team_projections.pop('minutes')
        team_performance.pop('minutes')
        return team_performance, team_projections

    def compare_players(self, player_A, player_B, n_recent_games):
        A_perf, A_proj = self.get_player_projection(player_A, n_recent_games)
        B_perf, B_proj = self.get_player_projection(player_B, n_recent_games)

        return (A_perf, A_proj), (B_perf, B_proj)
