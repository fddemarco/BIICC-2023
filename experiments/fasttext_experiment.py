import os
from glob import glob
import subprocess

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import fasttext

import dimension_generator as dg
from append_files import AppendFiles


class FasttextExperiment:
    def __init__(self, year, data_dir, texts_dir, results_dir, fasttext_pathname):
        self.year = year
        self.data_dir = data_dir
        self.texts_dir = texts_dir
        self.results_dir = results_dir
        self.fasttext_pathname = fasttext_pathname

    def run_experiment(self):
        self.generate_texts()
        self.run_fasttext()
        self.save_embeddings_to_csv()
        self.compare_rankings()

    def generate_texts(self):
        self.texts_by_subreddit()
        self.texts_to_single_file()

    def write_to_file(self, df_chunk):
        df = pd.DataFrame(df_chunk)
        df['text'] = self.texts_from(df)
        grouped = df.groupby('subreddit')['text'].apply(lambda x: ' '.join(x)).reset_index()
        for idx, row in grouped.iterrows():
            with open(self.subreddit_text_pathname(row['subreddit'], '.txt'), 'a') as f:
                f.write(row['text'])
        return df_chunk

    def texts_by_subreddit(self):
        data = load_dataset(
            'parquet',
            data_files=self.data_files_pathname(),
            split='train',
            streaming=True
        )
        data_mapped = data.map(self.write_to_file, batched=True, batch_size=10000)
        for data in data_mapped:
            pass

    def texts_to_single_file(self):
        files = self.every_subreddit()
        files.sort()
        with AppendFiles(files, self.subreddits_pathname('.txt')) as append_files:
            append_files.run()

    def run_fasttext(self):
        command = [self.fasttext_pathname, "skipgram", "-input", self.subreddits_pathname('.txt'),
                   "-output", self.subreddits_pathname(''),
                   "-epoch", "1", "-dim", "300", "-thread", "8"]
        result = subprocess.run(command)
        if result.returncode != 0:
            print("Command failed with error:")
            print(result.stderr)
            raise Exception

    def save_embeddings_to_csv(self):
        subreddits = [s for s in waller_ranking_arxiv()
                      if self.subreddit_text_exists(s)]
        embeddings = self.embeddings_of(subreddits)
        tf_idf = pd.DataFrame(embeddings, index=subreddits, columns=range(0, 300))
        tf_idf.to_csv(self.embedding_pathname())

    def embeddings_of(self, subreddits):
        model_pathname = self.subreddits_pathname('.bin')
        model = fasttext.load_model(model_pathname)
        embeddings = []
        for s in subreddits:
            pathname = self.subreddit_text_pathname(s, '.txt')
            with open(pathname, 'r') as f:  # TODO: Si el subreddit es muy grande, puede fallar
                text = f.read()
            embeddings.append(model.get_sentence_vector(text).astype(float))
        return embeddings

    def compare_rankings(self):
        fasttext_ranking = self.get_fasttext_ranking()
        waller_ranking = get_waller_ranking_for(fasttext_ranking)
        rankings = [
            {
                "Model": ["waller", "fasttext"],
                "Rank": [i + 1, fasttext_ranking.index(subreddit) + 1],
                "Subreddit": subreddit
            } for i, subreddit in enumerate(waller_ranking)
        ]
        bump_chart(rankings, len(waller_ranking), self.results_dir)

    def get_fasttext_ranking(self):
        df = pd.read_csv(self.embedding_pathname(), index_col=0)
        dimensions = dg.DimensionGenerator(df).generate_dimensions_from_seeds([("democrats", "Conservative")])
        scores = dg.score_embedding(df, zip(["dem_rep"], dimensions))
        fasttext_ranking = [x for x in scores.sort_values('dem_rep').index]
        return fasttext_ranking

    def subreddit_text_exists(self, s):
        return os.path.exists(self.subreddit_text_pathname(s, '.txt'))

    def every_subreddit(self):
        return glob(self.subreddit_text_pathname('', '*'))

    def data_files_pathname(self):
        return os.path.join(self.data_dir, f'{self.dataset()}_{self.year}-[0-9]*.parquet')

    def subreddit_text_pathname(self, subreddit, suffix):
        return os.path.join(self.texts_dir, f'subreddit_{subreddit}' + suffix)

    def subreddits_filename(self, extension):
        return f'subreddits_{self.year}' + extension

    def subreddits_pathname(self, extension):
        return os.path.join(self.results_dir, self.subreddits_filename(extension))

    def embedding_pathname(self):
        return os.path.join(self.results_dir, f'embeddings.csv')

    def dataset(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def texts_from(self, df):
        raise NotImplementedError("This method should be implemented in a subclass.")


class FasttextExperimentForComments(FasttextExperiment):
    def dataset(self):
        return 'RC'

    def texts_from(self, df):
        return df['body']


class FasttextExperimentForSubmissions(FasttextExperiment):
    def dataset(self):
        return 'RS'

    def texts_from(self, df):
        return df['title'] + ' ' + df['selftext']


def get_waller_ranking_for(ranking):
    return [s for s in waller_ranking_arxiv() if s in ranking]


def bump_chart(elements, n, results_dir):
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
        results_dir,
        f'rankings_comparison.png'),
        dpi=300,
        bbox_inches='tight')


def waller_ranking_arxiv():
    return [
        'democrats',
        'EnoughLibertarianSpam',
        'hillaryclinton',
        'progressive',
        'BlueMidterm2018',
        'EnoughHillHate',
        'Enough_Sanders_Spam',
        'badwomensanatomy',
        'racism',
        'GunsAreCool',
        'Christians',
        'The_Farage',
        'new_right',
        'conservatives',
        'metacanada',
        'Mr_Trump',
        'NoFapChristians',
        'TrueChristian',
        'The_Donald',
        'Conservative'
]




