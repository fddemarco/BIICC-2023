""" Dimension generator script used in Waller et al"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class DimensionGenerator:
    """Dimension Generator class"""

    def __init__(self, vectors):
        self.vectors = pd.DataFrame(
            normalize(vectors, norm="l2", axis=1), index=vectors.index
        )
        self.name_mapping = {name.lower(): name for name in vectors.index}
        self.directions_to_score = None

    def calculate_dimensions(self, nn_n=10):
        if self.directions_to_score is None:
            comm_names = list(self.vectors.index)
            cosine_sims = cosine_similarity(self.vectors)

            # Find each community's nearest neighbours

            np.fill_diagonal(cosine_sims, -10)
            rows_i = np.array([])
            cols_i = np.array([])

            total_dir = 0
            for i in range(0, len(cosine_sims)):
                v = cosine_sims[i].argsort().argsort()

                only_calculate_for = v > (len(comm_names) - nn_n - 2)
                total_dir = total_dir + np.sum(only_calculate_for)
                cols = np.nonzero(only_calculate_for)
                cols_i = np.append(cols_i, cols[0])
                rows_i = np.append(rows_i, [i] * len(cols[0]))
            indices_to_calc = (rows_i.astype(int), cols_i.astype(int))

            index = []
            directions = []
            for i in range(0, len(indices_to_calc[0])):
                c1 = indices_to_calc[0][i]
                c2 = indices_to_calc[1][i]
                index.append((comm_names[c1], comm_names[c2]))
                directions.append(self.vectors.iloc[c2] - self.vectors.iloc[c1])

            print(f"{total_dir} valid directions, {len(directions)} calculated.")
            self.directions_to_score = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(index), data=directions
            )
        return self.directions_to_score

    def generate_dimensions_from_seeds(self, seeds):
        """Generates multiple dimensions from seeds"""
        return list(map(lambda x: self.generate_dimension_from_single_seed([x]), seeds))

    def generate_dimension_from_single_seed(self, seeds):
        """Generates single dimension from seeds"""

        seed_directions = (
            self.vectors.loc[map(lambda x: x[1], seeds)].values
            - self.vectors.loc[map(lambda x: x[0], seeds)].values
        )

        seed_similarities = np.dot(self.calculate_dimensions(), seed_directions.T)
        seed_similarities = np.amax(seed_similarities, axis=1)

        directions = self.calculate_dimensions().iloc[
            np.flip(seed_similarities.T.argsort())
        ]

        # How many directions to take?
        num_directions = 10

        # make directions unique subreddits (subreddit can only occur once)
        ban_list = [s for sd in seeds for s in sd]
        i = -1  # to filter out seed pairs
        while (i < len(directions)) and (i < (num_directions + 1)):
            ban_list.extend(directions.index[i])

            l0 = directions.index.get_level_values(0)
            l1 = directions.index.get_level_values(1)
            directions = directions[
                (np.arange(0, len(directions)) <= i)
                | ((~l0.isin(ban_list)) & (~l1.isin(ban_list)))
            ]

            i += 1

        # Add seeds to the top
        directions = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(seeds + directions.index.tolist()),
            data=np.concatenate((seed_directions, directions.to_numpy())),
        )

        direction_group = directions.iloc[0:num_directions]

        dimension = np.sum(direction_group.values, axis=0)

        return {
            "note": "generated from seed pairs",
            "seed": seeds,
            "vector": dimension,
            "left_comms": list(map(lambda x: x[0], direction_group.index)),
            "right_comms": list(map(lambda x: x[1], direction_group.index)),
        }

    def get_scores_from_seeds(self, seeds, names):
        """Calculate score for embeddings over dimensions"""
        columns = {}

        dimensions = self.generate_dimensions_from_seeds(seeds)
        for name, data in zip(names, dimensions):
            columns[name] = np.dot(
                self.vectors.values, data["vector"] / np.linalg.norm(data["vector"])
            )

        return pd.DataFrame(columns, index=self.vectors.index)


import numpy as np
import pandas as pd

import os


def load_embedding():
    embedding_folder = os.path.join(os.path.dirname(__file__), "embedding")
    vectors_path = os.path.join(embedding_folder, "vectors.tsv")
    metadata_path = os.path.join(embedding_folder, "metadata.tsv")
    meta = pd.read_csv(metadata_path, sep="\t", header=None)

    meta.columns = meta.iloc[0]
    meta = meta.reindex(meta.index.drop(0))

    meta.set_index(meta.columns[0], inplace=True)

    vectors = pd.read_csv(vectors_path, sep="\t", header=None)
    vectors.set_index(meta.index, inplace=True)
    vectors = vectors.divide(np.linalg.norm(vectors.values, axis=1), axis="rows")

    return vectors, meta


def score_embedding(vectors, dimensions):
    columns = {}

    for name, data in dimensions:
        columns[name] = np.dot(
            vectors.values, data["vector"] / np.linalg.norm(data["vector"])
        )

    return pd.DataFrame(columns, index=vectors.index)


# cosine similarity of all vectors
def cosine_similarity(vectors):
    # normalize vectors
    vectors = vectors.divide(np.linalg.norm(vectors.values, axis=1), axis="rows")
    # dot
    sims = np.dot(vectors.values, vectors.values.T)

    return sims


class WallerDimenGenerator:
    def __init__(self, vectors):
        self.vectors = pd.DataFrame(
            normalize(vectors, norm="l2", axis=1), index=vectors.index
        )
        self.name_mapping = {name.lower(): name for name in vectors.index}

        comm_names = list(self.vectors.index)
        cosine_sims = cosine_similarity(self.vectors)

        # Find each community's nearest neighbours
        ranks = cosine_sims.argsort().argsort()

        # Take n NNs
        nn_n = 10
        only_calculate_for = (ranks > (len(comm_names) - nn_n - 2)) & ~np.diag(
            np.ones(len(comm_names), dtype=bool)
        )

        indices_to_calc = np.nonzero(only_calculate_for)

        index = []
        directions = []
        for i in range(0, len(indices_to_calc[0])):
            c1 = indices_to_calc[0][i]
            c2 = indices_to_calc[1][i]
            index.append((comm_names[c1], comm_names[c2]))
            directions.append(self.vectors.iloc[c2] - self.vectors.iloc[c1])

        print(
            "%d valid directions, %d calculated."
            % (np.sum(only_calculate_for), len(directions))
        )
        self.directions_to_score = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(index), data=directions
        )

    def generate_dimensions_from_seeds(self, seeds):
        return list(map(lambda x: self.generate_dimension_from_seeds([x]), seeds))

    def generate_dimension_from_seeds(self, seeds):
        seed_directions = (
            self.vectors.loc[map(lambda x: x[1], seeds)].values
            - self.vectors.loc[map(lambda x: x[0], seeds)].values
        )

        seed_similarities = np.dot(self.directions_to_score, seed_directions.T)
        seed_similarities = np.amax(seed_similarities, axis=1)

        directions = self.directions_to_score.iloc[
            np.flip(seed_similarities.T.argsort())
        ]

        # How many directions to take?
        num_directions = 10

        # make directions unique subreddits (subreddit can only occur once)
        ban_list = [s for sd in seeds for s in sd]
        i = -1  # to filter out seed pairs
        while (i < len(directions)) and (i < (num_directions + 1)):
            ban_list.extend(directions.index[i])

            l0 = directions.index.get_level_values(0)
            l1 = directions.index.get_level_values(1)
            directions = directions[
                (np.arange(0, len(directions)) <= i)
                | ((~l0.isin(ban_list)) & (~l1.isin(ban_list)))
            ]

            i += 1

        # Add seeds to the top
        directions = pd.concat([pd.DataFrame(seed_directions, index=seeds), directions])

        direction_group = directions.iloc[0:num_directions]

        dimension = np.sum(direction_group.values, axis=0)

        return {
            "note": "generated from seed pairs",
            "seed": seeds,
            "vector": dimension,
            "left_comms": list(map(lambda x: x[0], direction_group.index)),
            "right_comms": list(map(lambda x: x[1], direction_group.index)),
        }

    def get_scores(self, seeds, dimen_names):
        dimensions = self.generate_dimensions_from_seeds(seeds)
        return score_embedding(self.vectors, zip(dimen_names, dimensions))
