import os

import matplotlib.pyplot as plt
import pandas as pd
import fasttext

import dimension_generator as dg


class FasttextExperiment:
    def __init__(self, year, texts_dir, model_pathname, dataset, results_dir):
        self.model = fasttext.load_model(model_pathname)
        self.texts_dir = texts_dir
        self.year = year
        self.dataset = dataset
        self.results_dir = results_dir

    def subreddit_pathname(self, subreddit, suffix):
        return os.path.join(self.texts_dir, f'subreddit_{subreddit}' + suffix)

    def embeddings_of(self, subreddits):
        embeddings = []
        for s in subreddits:
            pathname = self.subreddit_pathname(s, '.txt')
            with open(pathname, 'r') as f:  # TODO: Si el subreddit es muy grande, puede fallar
                text = f.read()
            embeddings.append(self.model.get_sentence_vector(text).astype(float))
        return embeddings

    def save_embeddings_to_csv(self):
        selected = ['democrats', 'hillaryclinton', 'The_Donald', 'Conservative', 'Mr_Trump', 'BlueMidterm2018',
                    'EnoughHillHate', 'metacanada', 'TrueChristian', 'new_right',
                    'progressive', 'racism', 'GunsAreCool', 'Christians', 'The_Farage', 'NoFapChristians',
                    'Enough_Sanders_Spam', 'conservatives', 'Mr_Trump', 'EnoughLibertarianSpam', 'badwomensanatomy']

        selected_year = [s for s in selected if os.path.exists(self.subreddit_pathname(s, '.txt'))]
        print(selected_year)
        embeddings = self.embeddings_of(selected_year)
        tf_idf = pd.DataFrame(embeddings, index=selected_year, columns=range(0, 300))
        tf_idf.to_csv(self.embedding_pathname())

    def embedding_pathname(self):
        return os.path.join(self.results_dir, f'embeddings_{self.dataset}_{self.year}.csv')

    def scores_f(self):
        df = pd.read_csv(self.embedding_pathname(), index_col=0)
        dim = dg.DimensionGenerator(df)
        dimensions = dim.generate_dimensions_from_seeds([("democrats", "Conservative")])
        scores = dg.score_embedding(df, zip(["dem_rep"], dimensions))
        fasttext_ranking = [x for x in scores.sort_values('dem_rep').index]
        waller_ranking = [
            'democrats',
            'EnoughLibertarianSpam',
            'hillaryclinton',
            'progressive',
            'BlueMidterm2018',
            'Enough_Sanders_Spam',
            'badwomensanatomy',
            'racism',
            'GunsAreCool',
            'Christians',
            'The_Farage',
            'new_right',
            'conservatives',
            'metacanada',
            'NoFapChristians',
            'TrueChristian',
            'The_Donald',
            'Conservative'
        ]

        waller_ranking = [s for s in waller_ranking if s in fasttext_ranking]
        rankings = []
        for i, e in enumerate(waller_ranking):
            rankings.append(
                {"Model": ["waller", "fasttext"],
                 "Rank": [i + 1, fasttext_ranking.index(e) + 1], "Subreddit": e})
        self.bump_chart(rankings, len(waller_ranking))

    def bump_chart(self, elements, n):
        fig, ax = plt.subplots(figsize=(12, 6))
        for element in elements:
            ax.plot(
                element["Model"],
                element["Rank"],
                "o-",
                markerfacecolor="white",
                linewidth=3
            )
            ax.annotate(
                element["Subreddit"],
                xy=("fasttext", element["Rank"][1]),
                xytext=(1.01, element["Rank"][1])
            )
            ax.annotate(
                element["Subreddit"],
                xy=("waller", element["Rank"][0]),
                xytext=(-0.3, element["Rank"][0])
            )

        plt.gca().invert_yaxis()
        plt.yticks([i for i in range(1, n+1)])

        ax.set_xlabel('Model')
        ax.set_ylabel('Rank')
        ax.set_title('Comparison of Models on Subreddit Classification Task')

        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir,
            f'rankings_comparison_{self.year}.png'),
            dpi=300,
            bbox_inches='tight')
