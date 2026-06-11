
import numpy as np


def softmax(scores):
    # turn a row of raw scores into weights that are positive and sum to 1.
    # subtract the max first -> numerically safe (same result, no giant exp()).
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def attention(Q, K, V):
    # The whole attention mechanism, forward pass. Three steps:
    #
    # Args:
    #   Q : (n_queries, d_k)  -- one query vector per "asking" position
    #   K : (n_keys,    d_k)  -- one key   vector per position to look at
    #   V : (n_keys,    d_v)  -- one value vector per position to look at
    #
    # Returns:
    #   output  : (n_queries, d_v)        -- the blended values, per query
    #   weights : (n_queries, n_keys)     -- how much each query attended to each key

    d_k = Q.shape[-1]

    # STEP 1 -- match: dot every query against every key.
    #   Q @ K.T gives a score for each (query, key) pair. The "/ sqrt(d_k)"
    #   is the scaling housekeeping so scores don't blow up for big d_k.
    scores = (Q @ K.T) / np.sqrt(d_k)        # (n_queries, n_keys)

    # STEP 2 -- normalize: softmax each query's scores into attention weights.
    weights = softmax(scores)                # rows sum to 1

    # STEP 3 -- gather: weighted blend of the values, using those weights.
    output = weights @ V                     # (n_queries, d_v)

    return output, weights


if __name__ == "__main__":
    # Three "words" we could look at. Keys = how they get found.
    # Make them clearly distinct so the matching is easy to read.
    #            [animal, place, action]
    K = np.array([[1.0, 0.0, 0.0],   # cat    -> "i'm an animal"
                  [0.0, 1.0, 0.0],   # garden -> "i'm a place"
                  [0.0, 0.0, 1.0]])  # dog    -> "i'm an action-ish word"

    # Values = what each word actually hands over once found.
    V = np.array([[10.0, 0.0],       # cat's content
                  [20.0, 0.0],       # garden's content
                  [30.0, 0.0]])      # dog's content

    # One query from "was black": it's looking for an ANIMAL.
    Q = np.array([[1.0, 0.0, 0.0]])  # "show me animals"

    output, weights = attention(Q, K, V)

    print("attention weights (cat, garden, dog):")
    print(np.round(weights, 3))
    print("\nblended output (mostly cat's value [10, 0]):")
    print(np.round(output, 3))
