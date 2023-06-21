# Metricas

## TO DO LIST

* Capítulo 8.3.2.3 de Francesco Ricci, Lior Rokach, Bracha Shapira (eds.) - Recommender Systems Handbook-Springer US (2015)
  * Spearman
  * Kendall
  * Normalized Discounted Cumulative Gain (NDCG)
* <https://stats.stackexchange.com/questions/159657/metrics-for-evaluating-ranking-algorithms>
* <https://towardsdatascience.com/20-popular-machine-learning-metrics-part-2-ranking-statistical-metrics-22c3e5a937b6>
* <https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)>
* <https://towardsdatascience.com/how-to-objectively-compare-two-ranked-lists-in-python-b3d74e236f6a?>
  * Rank-biased overlap. <https://dl.acm.org/doi/10.1145/1852102.1852106> (Section 4.*)
  * Kendall Tau. <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>
  * Pearson's Correlation. <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>

## Métricas para rankings

Las medidas de evaluación para un sistema de recuperación de información (IR, por sus siglas en inglés) evalúan qué tan bien un índice, motor de búsqueda o base de datos devuelve resultados de una colección de recursos que satisfacen la consulta de un usuario. Por lo tanto, son fundamentales para el éxito de los sistemas de información y plataformas digitales. El éxito de un sistema de IR puede ser juzgado por una serie de criterios, incluyendo la relevancia, la velocidad, la satisfacción del usuario, la usabilidad, la eficiencia y la confiabilidad. Sin embargo, el factor más importante para determinar la eficacia de un sistema para los usuarios es la relevancia general de los resultados recuperados en respuesta a una consulta. Las medidas de evaluación pueden ser categorizadas de varias formas, incluyendo offline o en línea, basadas en el usuario o en el sistema, e incluyen métodos como el comportamiento observado del usuario, colecciones de pruebas, precisión y recuperación, y puntuaciones de conjuntos de pruebas de referencia preparados.

* <https://medium.com/nerd-for-tech/evaluating-recommender-systems-590a7b87afa5>

## Recommender Systems

Supose we want to measure the effectiveness of a recommendation system. It assigns a ranking 1-5 to every pair user-item, and then tries to make a good recommendation based on these guessed rankings. 

Precision and recall are binary metrics used to evaluate models with binary output (True/False). To measure the quality of our recommendation system, we will first translate our problem to a binary decision problem. To do the translation, we will assume that any *true rating* above 3.5 corresponds to a **relevant item**, and any *true rating* below 3.5 is **irrelevant**. Then, we can say that a *relevant item* for a specific user-item pair is a good recommendation for the user in question. Conversely, an *irrelevant item* would not be a good recommendation for the user.

In summary,

**Relevant** items are already known in the data set

* Relevant item: Has a True/Actual rating >= 3.5
* Irrelevant item: Has a True/Actual rating < 3.5

**Recommended** items are generated by recommendation algorithm

* Recommended item: has a predicted rating >= 3.5
* Not recommended item: Has a predicted rating < 3.5

### Precision and Recall

Terminology.

* **with condition**. It's the number of all possible relevant items for a user.
* **predicted positive**. It's the number of items we recommended.
* **correct positives**. It's the number of our recommendations that are actually relevant.

* Precision: P = $ \frac{\text{\# correct positive}}{\text{\# predicted positive}}$ = 1 - false positive rate

* Recall: R = $ \frac{\text{\# correct positive}}{\text{\# with condition}}$ = 1 - false negative rate

### Precision at k (P@k)

In the context of recommendation systems, supose we have 1000 items that we recommend. Clearly, it is not possible to show all 1000 items at the same time to the user. Instead, our focus is likely on recommending the top-N items (with N << 100).

Note that Precision and Recall metrics do not take ordering into account. To address this issue, we can evaluate these metrics at a specific cutoff point k, denoted P/R at k (or P/R@k). That is, we calculate the precision and recall by considering only recommendations from rank 1 through k. k is a user-definable integer set to match the top-k recommendations objective.

Precision at k is the proportion of *recommended items* in the top-k set that are relevant. That is,

Precision@k = (# of *recommended* items @k that are **relevant**) / (# of *recommended* items @k)

Fuente: <https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54>

For modern (web-scale) information retrieval, recall is no longer a meaningful metric, as many queries have thousands of relevant documents, and few users will be interested in reading all of them. Precision at k documents (P@k) is still a useful metric (e.g., P@10 or "Precision at 10" corresponds to the number of relevant results among the top 10 retrieved documents), but fails to take into account the positions of the relevant documents among the top k.[13]

Fuente: <https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_k>

### Average Precision

If we are asked to recommend N items, the number of relevant items in the full space of items is m, then:

$$
\text{AP@N} = \frac{1}{m}\sum_{k=1}^{N}{
\begin{cases}
 \text{P@k }& \text{if } item_k\text{ is relevant} \\
 0 & \text{otherwise,}
\end{cases}
}
$$
where

* N = # of recommended items
* M = # of all relevant items

Notice it is larger when there have been more successes in front of it - that's because the precision of the kth subset is higher the more correct guesses you've had up to point k. Thus, AP rewards you for front-loading the recommendations that are most likely to be correct.

## Average value of F1@k

F1-score (alternatively, F1-Measure), is a mixed metric that takes into account both Precision and Recall.

Similarly to Precision@k and Recall@k is a rank-based metric that can be summarized as follows: "What F1-score do I get if I only consider the top k predictions my model outputs?"

$$ F_1@k = 2\frac{P(k) \cdot R(k)}{P(k) + R(k)}$$

Fuente:
<https://queirozf.com/entries/evaluation-metrics-for-ranking-problems-introduction-and-examples#f1-k>

## CG (Cumulative Gain)

Cumulative Gain (CG) is the sum of the graded relevance values of all results in a search result list. The CG at a particular rank position $p$ is defined as:

$$ CG_p = \sum_{i=1}^p rel_i,$$
where $rel_i$ is the graded relevance of the recommendation at position $i$. If ratings are binary, $rel_i \in \{0, 1\}$. 

Note that the value computed with the CG function is unaffected by changes in the ordering of search results. That is, moving a highly relevant item upfront does not change the computed value for CG.

## DCG: (Discounted Cumulative Gain)

Three assumptions are made in using DCG and its related measures.

* Highly relevant documents are more useful when appearing earlier in a search engine result list (have higher ranks)
* Highly relevant documents are more useful than marginally relevant documents, which are in turn more useful than non-relevant documents.
* The user is expected to read a relatively large portion of the list.

Using a graded relevance scale of documents in a search-engine result set, DCG measures the usefulness, or gain, of a document based on its position in the result list. The gain is accumulated from the top of the result list to the bottom, with the gain of each result discounted at lower ranks.

In the following, $rel_i$ is the graded relevance of the result at position $i$.

Como mi sistema no tiene relevance score para las contranarrativas, hay que fijarles un valor constante (se puede fijar valor 1) (Here we assume that the relevance score of each document to a query is given (otherwise it is usually set to a constant value), esto lo dice acá:
<https://towardsdatascience.com/20-popular-machine-learning-metrics-part-2-ranking-statistical-metrics-22c3e5a937b6)>.

The traditional formula of DCG accumulated at a particular rank position p is defined as:

$$ \text{DCG}_p = \sum_{i=1}^{p}{\frac{rel_i}{log_2(i+1)}}. $$

The premise of DCG is that highly relevant documents appearing lower in a search result list should be penalized as the graded relevance value is reduced logarithmically proportional to the position of the result.

Previously there was no theoretically sound justification for using a logarithmic reduction factor[2] other than the fact that it produces a smooth reduction. But Wang et al. (2013) gave theoretical guarantee for using the logarithmic reduction factor in Normalized DCG (NDCG). The authors show that for every pair of substantially different ranking functions, the NDCG can decide which one is better in a consistent manner.

An alternative formulation of DCG[4] places stronger emphasis on recommending relevant items:

$$ \text{DCG}_p = \sum_{i=1}^{p}\frac{2^{rel_i}-1}{log_2(i+1)}$$

The latter formula is commonly used in industry including major web search companies[5] and data science competition platforms such as Kaggle.[6]

These two formulations of DCG are the same when the relevance values of documents are binary;$ rel_i \in \{0,1\}$.

Note that Croft et al. (2010) and Burges et al. (2005) present the second DCG with a log of base e, while both versions of DCG above use a log of base 2. When computing NDCG with the first formulation of DCG, the base of the log does not matter, but the base of the log does affect the value of NDCG for the second formulation. Clearly, the base of the log affects the value of DCG in both formulations.

## Normalized DCG

Search result lists vary in length depending on the query. Comparing a search engine's performance from one query to the next cannot be consistently achieved using DCG alone, so the cumulative gain at each position for a chosen value of p should be normalized across queries. This is done by sorting all relevant documents in the corpus by their relative relevance, producing the maximum possible DCG through position p, also called Ideal DCG (IDCG) through that position. 

For each query, the nDCG is computed as:

$$ nDCG_k = \frac{DCG_k}{IDCG_k} $$
$$ IDCG_k = \sum_{i = 1}^{|REL_k|}{\frac{rel_i}{log_2(i+1)}}$$
where $REL_k$ represents the list of relevant items (ordered by their relevance) up to position $k$.

The nDCG values for all queries can be averaged to obtain a measure of the average performance of a search engine's ranking algorithm. 

Fuente: <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>

* Normalized DCG metric does not penalize for bad documents in the result.
* Normalized DCG does not penalize for missing documents in the result.
* Normalized DCG may not be suitable to measure performance of queries that may often have several equally good results. 

## Rank Biased Overlap

Why do you need a ranked list comparison? In fact, we all do such comparisons all the time. How to compare the ranked outputs generated by two different learning-to-rank / machine-learned ranking (MLR) models? How similar are two paragraphs once they are converted to tokenized word lists?

Then, how are you supposed to measure the quality of the sequences? You need some kind of measure. Ideally, a comparison measure should be in the form of a score that indicates how similar the rankings are.

* The ranked lists may be indefinite and possibly of different lengths — hence the measure should be able to handle two different list sizes.
* There should be an ability to compare at a chosen intersecting length — which means the measure should handle the intersection length of the lists.
* The top elements have more importance (like in search results, movie order, etc.) than the bottom ones — hence there should be a possibility to give weights whenever needed.

The Rank Biased Overlap (RBO) is a rank-biased method, as its name implies. 

We define the intersection of two lists at depth (d) as follows.
$$ I_d = S_d \cap T_d. $$

Consider S = [1,2,3,4,5,6,7] and T = [1,3,2,4,5,7,6,8]. Then, 
$I_d = [1,2,3,4,5,6,7]$. The depth is taken as at most the length of the smaller list. The length of this intersection is called the **Overlap**, which in this case is 7. We denote the Overlap as $X_d = |I_d|$. We define the **Agreement** as the proportion of S and T that are overlapped at depth $d$:
$$ A_d = \frac{X_d}{d}. $$
Note that $0 \leq X_d \leq d$; thus, $A_d \in [0, 1]$.

The **Average Overlap** (AO) is the average agreement for all the depths of the lists, ranging from 1 to an integer k.
$$ AO = \frac{1}{k} \sum_{d=1}^{k}{A_d}.$$
This is the foundation for the similarity measures. We can then define a family of similarity measures (SIM) as follows:
$$ \text{SIM}_w = \sum_{d=1}^{\infty} w_d A_d.$$
Here, $w$ represents the weight vector, and $w_d$ represents the weight at $d$.

Note that if the series of weights is convergent, that is, the sum of weights is bounded; thus, the SIM is also bounded. One such convergent series is the geometric progression $w = 1, p, p^2, ...$, where the $d$-th term has the value $p^{d-1}$. 

Recall that the infinite sum of a geometric progression $a_i = a_1 \cdot r^{i-1}$ 
$$ S = \sum_{i=1}^{\infty} a_i = \frac{a_1}{1 - r}, |r| < 1.$$

Assumming that $0 < p < 1$, then:
$$ \sum_{d=1}^{\infty} p^{d-1} = \frac{1}{1-p}.$$
The **Rank Biased Overlap** is a particular instance of this family of measures, where 
$$w_d = (1 - p) p^{d-1}.$$
That is,
$$\text{RBO}_p = (1 - p)\sum_{d=1}^{\infty}{p^{d-1}A_d}.$$
where $0 < p < 1$.

RBO falls in the range $[0,1]$, where 0 means disjoint, and 1 means identical. The argument $p$ determines how steep is the decline in weights: the smaller p, the more top-weighted is the metric. 

RBO is defined on infinite lists. Since it is convergent, evaluating a prefix establishes both a minimum and a maximum score for the full evaluation. The range between these two scores represents the remaining uncertainty associated with the prefix evaluation, as opposed to the full evaluation.

To set a minimum on the full evaluation, we can modify RBO's formula to prefix at depth $k$, which is usually referred to as RBO@k. This approach sets a lower bound on the full evaluation. However, we can improve this bound even further as follows.

Whenever $k < d$, $I_k \subseteq I_d$; thus, $X_k \leq X_d$ and $A_d \geq X_k/d$. Then, we can assure that the sum of the agreements at depths beyond $k$ is greater or equal than:
$$ (1-p) \sum_{d=k+1}^{\infty}{\frac{X_k}{d}p^{d-1}}. $$

To set a tight bound on full evaluation, add this term to the RBO@k score. The infinite sum can be resolved to finite form by the equality (Taylor's series of -ln(1-x)):
$$ \sum_{i=1}^{\infty}\frac{p^i}{i} = \ln \frac{1}{1-p}, 0 < p < 1.$$

After some rearrangement, 

$$ RBO_{MIN}(p,k) = \frac{1-p}{p} \left(\sum_{d=1}^{k}{(X_d - X_k)\frac{p^d}{d} - X_k \ln(1-p)}\right)$$

Prefix evaluation can also be used to derive a tight upper bound on the full RBO score; the residual uncertainty of the evaluation is then the difference between the lowe and upper bounds. The maximum score occurs when every element past prefix depth k in each ranking matches an element in the other ranking.

At each successive depth, two more elements are taken into consideration, one from each ranking. Therefore, the upper bound overlap increases by two until agreement is complete, which occurs at depth $f = 2k - X_k$. Beyond this depth, agreement is fixed at 1. Therefore, the residual RBO value is:

$$ RBO_{RES}(p,k) = (1-p) \left( \sum_{d=k+1}^{f}{\frac{2(d-k)}{d}p^{d-1} + \sum_{d=f+1}^{\infty}{\left(1-\frac{X_k}{d}\right)p^{d-1}}}\right)$$

### Rank Weights under RBO

Consider the difference in the final score between, on the one hand, both elements at depth d being matched at or prior to depth d (maximum agreement), and, on the other, neither element being matched at infinite depth (minimum agreement). We will refer to this difference as the weight of rank d, denoted as $W_{RBO}(d)$.

$$W_{RBO}(d) = \frac{1-p}{p}\sum_{i=d}^{\infty}{\frac{p^i}{i}}.$$

The weight of the prefix of length d, $W_{RBO}(1:d)$, is the sum of the weights of the ranks up to that depth.
$$ W_{RBO}(1:d) = 1 - p^{d-1} + d \frac{1-p}{p} \left( \ln \frac{1}{1-p} - \sum_{i=1}^{d-1}{\frac{p^i}{i}}\right).$$

Thus, the experimenter can tune the metric to achieve a given weight for a certain length of prefix.

### Extrapolation

Definitions of RBO MIN and RBO RES have been formulated in Section 4.2. The RBO score can then be quoted either as base+residual or as a min–max range. For many practical and statistical applications, though, it is desirable or necessary to have a single score or point estimate, rather than a range of values.

The simplest method is to use the base RBO value as the single score for the partial evaluation. The base score gives the known similarity between the two lists, the most that can be said with certainty given the information available.

An alternative formulation for a single RBO score is to extrapolate from the visible lists, assuming that the degree of agreement seen up to depth k is continued indefinitely.

Extrapolation assumes that the degree of agreement seen at k is expected to continue to higher ranks, that is, that for $r > k, A_r = X_k /k > X_k/r$.

### Metricity

Since RBO measures similarity, not distance, it is not a metric. However, RBO can be trivially turned into a distance measure, rank-biased distance (RBD), by RBD = 1 − RBO. In this context, *metric* is a symmetric measure where the triangle inequality holds.

Fuentes

* RBO. <https://towardsdatascience.com/how-to-objectively-compare-two-ranked-lists-in-python-b3d74e236f6a>
* Serie geométrica. <https://es.wikipedia.org/wiki/Progresi%C3%B3n_geom%C3%A9trica>
* Serie de Taylor. <https://es.wikipedia.org/wiki/Serie_de_Taylor#Logaritmo_natural>
* Paper original de RBO. <https://dl.acm.org/doi/10.1145/1852102.1852106> (Section 4.*)

## Using a Reference Ranking

### Fraction of Concordan Pairs (FCP) - Kendall tau

The Fraction of Concordant Pairs (FCP) is a metric used t evaluate the performance of ranking algorithms. It is often used in information retrieval and is a measure of the agreement between the orderings produced by two rankers for the same set of items.

The FCP metric is based on pairs of items and their corresponding rankings. Given two rankings, the first step is to consider all possible pairs of items that appear in both rankings. For each such pair, we determine whether the relative order of the two items is the same in both rankings (i.e., they are concordant) or whether their order is different (i.e., they are discordant).

The FCP is then calculated as the fraction of pairs that are concordant, or equivalently, the fraction of pairs for which the order is the same in both rankings. A perfect agreement between the two rankings would result in an FCP value of 1, whereas a completely random ranking would result in an FCP value of 0.5. An FCP value less than 0.5 indicates that the two rankings are more different than would be expected by chance.

The FCP metric is a useful tool for evaluating the agreement between different rankers, and it is often used in applications where it is important to have a consensus ranking.