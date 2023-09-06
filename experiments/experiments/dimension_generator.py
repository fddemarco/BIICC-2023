"""
    This module generates d-ness scores from given dimension seed pairs
"""
from typing import TypeAlias, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

SeedPair: TypeAlias = Tuple[str, str]
Dimension: TypeAlias = npt.NDArray[np.floating]


def similarity_matrix(vectors: pd.DataFrame) -> npt.NDArray[np.floating]:
    """Compute the cosine similarity pairwise between each row in vectors,
    but replace the principal diagonal with -inf.

    Args:
        vectors (pd.DataFrame): Input data.

    Returns:
        npt.NDArray[np.floating]: Cosine similarity matrix.
    """
    cosine_sims = cosine_similarity(vectors)
    np.fill_diagonal(cosine_sims, float("-inf"))
    return cosine_sims


class DimensionGenerator:
    """A class to generate d-ness scores from seed pairs."""

    def __init__(self, vectors: pd.DataFrame, nn_n: int = 10, k: int = 10):
        """Initialize a d-ness score generator with input data.

        Args:
            vectors (pd.DataFrame): Input data.
            nn_n (int, optional): Nearest Neighbours Number of pairs to generate per community.
            Defaults to 10.
            k (int, optional): Number of directions used to create the dimension.
            Defaults to 10.
        """
        self.vectors = pd.DataFrame(
            normalize(vectors, norm="l2", axis=1), index=vectors.index
        )
        self.nn_n = min(len(vectors), nn_n)
        self.k = k

    def nearest_neighbours_directions(self) -> pd.DataFrame:
        """This is based on the aforementioned idea that we are looking for
        pairs of communities that are very similar, but differ only in the target concept.

        Returns:
            pd.DataFrame: the set of all pairs of communities (c1, c2) such that c1 != c2
            and c2 is one of the nn_n nearest neighbours to c1. (nn_n = 10 by default)
        """
        comm_names = list(self.vectors.index)
        matrix = similarity_matrix(self.vectors)

        kth_largest_values = np.partition(matrix, -self.nn_n, axis=1)[:, -self.nn_n]
        indices_to_calc = np.where(matrix >= kth_largest_values[:, np.newaxis])

        pair_names = [
            (comm_names[c1], comm_names[c2]) for c1, c2 in zip(*indices_to_calc)
        ]
        pairs_difference = [
            self.vectors.iloc[c2] - self.vectors.iloc[c1]
            for c1, c2 in zip(*indices_to_calc)
        ]

        return pd.DataFrame(
            index=pd.MultiIndex.from_tuples(pair_names), data=pairs_difference
        )

    def generate_dimensions_from_seeds(
        self, seeds: Sequence[SeedPair]
    ) -> List[Dimension]:
        """Apply seed augmentation to all dimension seed pairs.

        Args:
            seeds (Sequence[SeedPair]): List of dimension seed pairs.

        Returns:
            List[Dimension]: List of augmented dimension representation.
        """
        return [self.augment_seed_direction(x) for x in seeds]

    def augment_seed_direction(self, seed_pair: SeedPair) -> Dimension:
        """Augment seed pair direction for a more robust representation of the dimension.

        All pairs are ranked based on the cosine similarity of their vector difference
        with the vector difference of the seed pair: cos(s2-s1, c2-c1). Additional pairs
        are then selected greedily.

        Args:
            seed_pair (SeedPair): Dimension seed pair.

        Returns:
            Dimension: Augmented dimension representation.
        """

        # 1-D Vector
        seed_direction = (
            self.vectors.loc[seed_pair[1]].to_numpy()
            - self.vectors.loc[seed_pair[0]].to_numpy()
        )
        nn_directions = self.nearest_neighbours_directions()

        # 1-D Vector. No me queda claro por que hace producto interno en vez de cosine similarity
        # los vectores no estan normalizados, asi que no son equivalentes
        seed_similarities = np.dot(nn_directions, seed_direction.T)

        # assert (seed_similarities >= -1).all()
        # assert (seed_similarities <= 1).all()

        directions = nn_directions.iloc[
            seed_similarities.argsort()[
                ::-1
            ]  # Sort DESC nearest neighbours by similarity
        ]

        directions = self.augmentation_algorithm(seed_pair, seed_direction, directions)
        return np.sum(directions.to_numpy(), axis=0)

    def augmentation_algorithm(
        self, seed_pair: SeedPair, seed_direction: np.array, directions: pd.DataFrame
    ) -> pd.DataFrame:
        """Seed augmentation algorithm.

        The most similar pair to the original seed pair
        that has no overlap in communities with the seed pair or any of the previously
        selected pairs is selected, and this process is repeated until k - 1
        additional pairs are selected, which results in the k pairs used to create
        the dimension.

        Args:
            seed_pair (SeedPair): Dimension seed pair.
            seed_direction (np.array): Seed pair direction.
            directions (pd.DataFrame): DESC Sorted nearest neighbours directions.

        Returns:
            pd.DataFrame: Augmented seed pair dimension directions.
        """
        ban_list = list(seed_pair)

        # Este algoritmo hay que revisarlo.
        i = -1  # filter out seed pair
        while i < len(directions) and i < self.k + 1:
            ban_list.extend(directions.index[i])

            c1 = directions.index.get_level_values(0)
            c2 = directions.index.get_level_values(1)
            directions = directions[
                (np.arange(0, len(directions)) <= i)
                | ((~c1.isin(ban_list)) & (~c2.isin(ban_list)))
            ]

            i += 1

        directions = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([seed_pair] + directions.index.tolist()),
            data=np.concatenate([[seed_direction], directions.to_numpy()]),
        )
        return directions.iloc[0 : self.k]

    def get_scores_from_seeds(self, seeds: List[SeedPair], names: List[str]):
        """Calculate dimensions scores.

        Args:
            seeds (List[SeedPair]): List of dimension seed pairs.
            names (List[str]): List of dimension names.

        Returns:
            pd.DataFrame: Dimensions scores DataFrame.
        """
        columns = {}

        dimensions = self.generate_dimensions_from_seeds(seeds)
        for name, dimension in zip(names, dimensions):
            columns[name] = np.dot(
                self.vectors.to_numpy(), dimension / np.linalg.norm(dimension)
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
