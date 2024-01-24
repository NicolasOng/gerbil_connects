# gerbil_connects

This repo contains repos for various end to end entity linking systems in the literature. They've been modified to run in a unified evaluation environment, and settings for candidate set ablation experiments have been created.

## Repo Overview
Here is a list of the repos where the experiments were successfully run. We will be adding the running instructions in their respective readme.md files.
- ebert (Poerner)
- efficient_bert_ent_emb (Feng)
- efficient-autoregressive-EL (De Cao)
- end2end_neural_el (Kolitsas)
- EntQA (Zhang)
- GENRE (De Cao)
- kb (Peters)
- REL (van Hulst)
- SpEL (Shavarani)

Here are the repos where experiments were attempted
- CHOLAN (Kannan Ravi. et al): Model not provided, training code not provided, data used in scripts missing
- entity_knowledge_in_bert (Broscheit): Model not provided, currently in training.

Other folders:
- empirical errors: The code for the error charts
- paper: Mostly for producing graphs that didn't make it into the paper
- test_servers: For some testing with the gerbil_connects framework

## Reproduction in a Unified Environment


Numbers are for Micro F1.
| Model | our test a | our test b | our test c | reported test a | reported test b | a diff | b diff |
| - | - | - | - | - | - | - | - |
| Kolitsas et al. (2018) | 89.50 | 82.44 | 65.75 | 89.4 | 82.4 | +0.1 | +0.04 |
| Peters et al. (2019) KnowBert-Wiki | 76.74 | 71.68 | 54.12 | 80.2 | 74.4 | -3.46 | -2.72 |
| Peters et al. (2019) KnowBert-W+W | 77.19 | 71.69 | 53.92 | 82.1 | 73.7 | -4.91 | -2.01 |
| Poerner et al. (2019) | 89.40 | 84.83 | 65.93 | 90.8 | 85.0 | -1.4 | -0.17 |
| van Hulst et al. (2020) Wiki 2014 | 83.30 | 82.53 | 71.69 | - | 83.3 | - | -0.77 |
| van Hulst et al. (2020) Wiki 2019 | 79.64 | 80.10 | 73.54 | - | 80.5 | - | -0.4 |
| De Cao et al. (2020) | 90.09 | 82.78 | 75.60 | - | 83.7 | - | -0.92 |
| De Cao et al. (2021) | 87.29 | 85.65 | 47.54 | - | 85.5 | - | +0.15 |
| Zhang et al. (2021) | 86.81 | 84.30 | 72.55 | - | 85.8 | - | -1.5 |
| Feng et al. (2022) | 87.64 | 86.49 | 65.05 | - | 86.3 | - | +0.19 |
| Shavarani and Sarkar (2023) large-500K (no cnds.) | 89.72 | 82.25 | 77.54 | 89.7 | 82.2 | +0.02 | +0.05 |
| Shavarani and Sarkar (2023) large-500K (Kb+Yago) | 89.89 | 82.88 | 59.50 | 89.8 | 82.8 | +0.09 | +0.08 |
| Shavarani and Sarkar (2023) large-500K (PPRforNED) | 91.58 | 85.22 | 46.98 | 91.5 | 85.2 | +0.08 | +0.02 |

### Unified Environment Description

The unified evaluation environment uses gerbil_connect to connect the models to the existing framework GERBIL.

In this evaluation environment, models are given plain text (eg: "The quick fox...") and are expected to output the annotations of that text as a list of character level annotations (eg: [(4, 9, "https://en.wikipedia.org/wiki/Quick"), (10, 13, "https://en.wikipedia.org/wiki/Fox"), ...]).

A unified environment was essential for running the candidate set ablation experiments.

### Changes made to the models

Unfortunately, many of the models do not follow this evaluation pattern in their evaluation code. Here are some common patterns that needed to be altered.

#### Modification 1 (Tokenization Step)
Reading directly from the AIDA dataset for evaluation input, instead of being given a string. This effectively tokenizes the input for the model in a way helpful for entity linking on AIDA. For example, "A. Smith, considered surplus to England's one-day requirements, struck 158, his first championship." looks like "A. Smith , considered surplus to England 's one-day requirements , struck 158 , his first championship ." in the AIDA file, where each space is a new line. We found that using nltk's punkt word tokenization utility worked well to turn text into tokens these models expect. Compared to splitting by whitespaces, punkt tokenization increases scores by ~15-20%.

#### Modification 2 (Document-Splitting Strategy)
Some models had document-splitting strategies (and reconsolidation strategies for the annotations) as the models have length limitations. We had to move them out of the model and after the new tokenization step for the relevant models in modification 1. For these models, no reconsolidation strategy existed, so we created our own.

Other models (GENRE) used a document splitting strategy for evaluation, but this strategy wasn't mentioned in the paper or is in the repo (See issue #30 in GENRE's github). For this, we used the splitting strategy created by Elevant who were able to replicate GENRE. Using this method instead of no splitting method increased the score by ~15%.

#### Modification 3 (Token-to-Character Annotation Conversion)
Some models don't make character-level predictions. Instead, they evaluate on their token-level predictions. For these models, we had to convert back from token predictions to character predictions.

#### "Modification" 4 (Outside data)
Generally, the data needed for reproduction are available to be downloaded. But for some models (GENRE), it had to be recreated. GENRE uses a modified mention-to-candidate dictionary for evaluation, which is breifly mentioned in Appendix A.2 under Setting. The modifications done to the dictionary are described in issues 30 & 37 in GENRE's repo, but it unfortunately wasn't released. To fix this, we used the dictionary created by Elevant, who were able to replicate GENRE. We found that using this dictionary instead of the usual one results in a +7% score.

#### "Modification" 5 (Training a new model)
Some models weren't available to us, but the training code was. We were able to use this to train new models that performed as well as the ones used for the paper.

These are the main changes we needed to make to fit the models into the unified evaluation environment. There are many other small things we needed to do to integrate the models into gerbil_connect, like changing the output format slightly, formatting and giving the model the data, etc.

Here is a table for a high level overview of the changes needed:

| Model | Changes |
| - | - |
| Kolitsas et al. (2018) | - |
| Peters et al. (2019) | 1, 2, 3 |
| Poerner et al. (2019) | 1, 2, 3, 5 |
| van Hulst et al. (2020) | - |
| De Cao et al. (2020) | 2, 4 |
| De Cao et al. (2021) | - |
| Zhang et al. (2021) | - |
| Feng et al. (2022) | 1, 2, 3, 5 |
| Shavarani and Sarkar (2023) | - |

## Candidate Set Ablations

For the candidate set ablation experiments, we did the following:

- Default Setting: Use the model's no-candidate-sets setting, if one was available.
- Full Setting: If one wasn't available, alter the model to use the in-domain mention vocabulary of AIDA as their candidate sets (about 5K entities)

Some experimental notes:
- We also had an "Empty Setting", where we altered the candidate sets to be empty. We ran this on nearly all the models, and they all got 0.
- We were planning to modify the Full Setting to include 500K or even 6M entities, but we decided not to based on the scores and time from the 5K experiments with the models.
- We also ran the "Full Setting" on nearly all the models (eg: GENRE's trie restricts entity generation to 5K entities instead of no restrictions). As mentioned in the paper, we didn't report the results if the model already had a default setting or didn't do well enough. Those "didn't do well enough" scores are added below.

| Model | test a | test b | test c | setting |
| - | - | - | - | - |
| De Cao et al. (2020) | 85.15 | 78.98 | 75.62 | Default |
| De Cao et al. (2021) | 62.00 | 49.51 | 37.05 | Default |
| Zhang et al. (2021) | 86.81 | 84.30 | 72.55 | Default |
| Shavarani and Sarkar (2023) | 89.72 | 82.25 | 77.54 | Default |
| Poerner et al. (2019) | 22.81 | 18.81 | 17.56 | Full |
| Feng et al. (2022) | 35.00 | 32.58 | 27.48 | Full |
| Kolitsas et al. (2018) | 0.04 | 0.22 | 0 | Full |
| Peters et al. (2019) KnowBert-Wiki | 1.05 | 0.49 | 0.24 | Full |
| Peters et al. (2019) KnowBert-W+W | 0.96 | 0.45 | 0.14 | Full |
| van Hulst et al. (2020) Wiki 2014 | 0.04 | 0.11 | 0 | Full |
| van Hulst et al. (2020) Wiki 2019 | 0.02 | 0.02 | 0 | Full |

