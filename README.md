# Skip-gram Word2vec implementation for JetBrains internship application

A week-long project for 2026 JetBrains internship application.
The aim was to implement word2vec without using *any* ML framework,
but to code it up from scratch using NumPy instead.

## What is it?
### word2vec
word2vec intends to capture a semantic meaning of words by embedding them in a learnt
(embedding) vector space, using for that purpose the notion of *similarity* between the words in the context.
### Continuous Skip-gram Model
It tries to maximize classification of a word based on another word in the same sentence.
    We use each current word as an input to a log-linear classifier with continuous
    projection layer, and predict words within a certain range before and after the current word
### Negative sampling
To avoid computing softmax over the entire vocabulary, we perform negative sampling.
    Namely, we consider the true word-context pair to be a "positive sample", and then
    choose to sample k random words that we will treat as "negative samples".
    

It is a surrogate objective: instead of maximizing the exact conditional probability of the context word
    under a normalized model,
    it learns embeddings that are good at discriminating true word-context co-occurrences from noise

#### Distribution
We use unigram raised to the power of 3/4 as a distribution for the negative sampling

### Subsampling
We use subsampling to limit the consideration of the most frequent words, as they
    provide less information than rare words (think "the" vs. "Warsaw", the pair ("Warsaw", "Poland") is
    more semantically informative than the pair ("Warsaw", "the").

The new sampling probability becomes:

            P(w_i) = sqrt(t/frequency(w_i)),
            where t is an empirically chosen hyperparameter threshold, typically around 1e-5

### Loss
Our goal is to maximize 

        log(sigm(u_o^T * v_c) + sum_i=1_k log(sigm(-u_n_i^T * v_c))
        
Therefore, the loss function is: L = - objective

Let us introduce s^+ := u_o^T * v_c, s_i^- := u_n_i^T * v_c
        
Since:

            -log(sigm(x)) = log(1+e^(-x))
            -log(sigm(-x)) = log(1+e^x)
We obtain:

            L = log(1+e^(-s^+)) + sum_i=1_k log(1+e^(s_i^-))
### Gradient
Since:

            d/dx (-log(sigm(x))) = sigm(x) - 1
            d/dx (-log(sigm(-x))) = sigm(x)
We obtain:

            dL/ds^+ = sigm(s^+) - 1
            dL/ds^- = sigm(-s_i^-)
Therefore, knowing that ds/dv_c = u:

            dL/dv_i = (sigm(s^+) - 1) * u_o + sum_i=1_k sigm(s_i^-) * u_n_i
            dL/du_o = (sigm(s^+) - 1) * v_c
            dL/du_n_i = sigm(s_i^-) * v_c
Thus, we obtain our gradient.

The update rule is:

            v_c <- v_c - lr * dL/dv_c
            u_o <- u_o - lr * dL/du_o
            u_n_i <- u_n_i - lr * dL/du_n_i

## Results

## Possible improvements
### Results
- Due to the time constraints, I was unable to provide a graph that would display loss across the training process

## References
- T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient Estimation of Word Representations in Vector Space,” arXiv.org, Sep. 06, 2013. http://arxiv.org/abs/1301.3781
- Y. Goldberg and O. Levy, “word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method,” arXiv.org, Feb. 15, 2014. http://arxiv.org/abs/1402.3722
- T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, “Distributed representations of words and phrases and their compositionality,” in Advances in Neural Information Processing Systems, C. J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, Eds., Curran Associates, Inc., 2013. Available: https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf