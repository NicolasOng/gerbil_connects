from typing import List, Set, Tuple, Dict

import torch
import numpy as np

from allennlp.data import DatasetReader, Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, TextField, LabelField, SpanField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.file_utils import cached_path

from kb.wiki_linking_util import WikiCandidateMentionGenerator, MentionGenerator
from kb.entity_linking import TokenCharactersIndexerTokenizer
from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator


INDEXER_DEFAULT = {
    "type": "characters_tokenizer",
    "tokenizer": {
        "type": "word",
            "word_splitter": {"type": "just_spaces"},
        },
        "namespace": "entity"
    }


@DatasetReader.register("aida_wiki_linking")
class LinkingReader(DatasetReader):

    """
    Reads entity linking data with the following format:

    boycott
    MMSTART_31717 TAB United_Kingdom
    British
    MMEND
    lamb
    .
    *NL*

    I.e one word per line, with `MMSTART_{wiki_id}` denoting the begining of entities and
    *NL* denoting new line boundaries.

    Documents are separated with:

    DOCSTART_4_CHINA
    ...
    DOCEND
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 entity_indexer: TokenIndexer = TokenIndexer.from_params(Params(INDEXER_DEFAULT)),
                 granularity: str = "sentence",
                 mention_generator: MentionGenerator = None,
                 should_remap_span_indices: bool = True,
                 entity_disambiguation_only: bool = False,
                 extra_candidate_generators: Dict[str, MentionGenerator] = None):

        lazy = False
        super().__init__(lazy)
        self.token_indexers = token_indexers or {"token": SingleIdTokenIndexer("token")}
        self.entity_indexer = {"ids": entity_indexer}
        self.separator = {"*NL*"}
        if granularity == "sentence":
            self.separator.add(".")

        if granularity not in {"sentence", "paragraph"}:
            raise ConfigurationError("Valid arguments for granularity are 'sentence' or 'paragraph'.")

        self.entity_disambiguation_only = entity_disambiguation_only
        self.mention_generator = mention_generator or WikiCandidateMentionGenerator()
        self.should_remap_span_indices = should_remap_span_indices

        self.extra_candidate_generators = extra_candidate_generators
    
    def set_no_candidate_sets(self):
        self.mention_generator.no_candidate_sets = True
        for _, generator in self.extra_candidate_generators.items():
            generator.no_candidate_sets = True
    
    def set_full_candidate_sets(self):
        self.mention_generator.full_candidate_sets = True
        for _, generator in self.extra_candidate_generators.items():
            # haven't made the full wordnet candidate list.
            generator.no_candidate_sets = True

    def _read(self, file_path: str):

        file_path = cached_path(file_path)
        words = []
        gold_spans = []
        gold_entities = []
        in_mention = False
        doc_id = None
        with open(file_path) as input_file:
            for line in input_file:
                line = line.rstrip()
                if line in self.separator and not in_mention:
                    if line == ".":
                        words.append(line)
                    # if we have continuous *NL* *NL* do not return empty chunks
                    if len(words) > 0:
                        processed = self.mention_generator.get_mentions_with_gold(" ".join(words), gold_spans,
                                                                                  gold_entities, whitespace_tokenize=True, keep_gold_only=self.entity_disambiguation_only)
                        if processed["candidate_spans"]:
                            yield self.text_to_instance(doc_id=doc_id, **processed)
                    # Reset state
                    words = []
                    gold_spans = []
                    gold_entities = []

                elif line.startswith('MMSTART_'):
                    in_mention = True
                    _, name = line.split("\t")
                    name = name.strip()
                    gold_entities.append(name)
                    gold_spans.append([len(words)])
                elif line == 'MMEND':
                    in_mention = False
                    # Spans are inclusive in allennlp
                    gold_spans[-1].append(len(words) - 1)
                elif line.startswith('DOCSTART_'):
                    # ignore doc indicators
                    doc_id = line.strip()
                elif line.startswith('DOCEND'):
                    doc_id = None
                else:
                    words.append(line)
        if words:
            # an instance for the instance list returned by read() is returned after each separator in the text file. (or at the end).
            # the input to get_mentions_with_gold looks like this:
            # "CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY ."
            # [(2, 2)]
            # ["Leicestershire_County_Cricket_Club"]
            # get_mentions_with_gold is defined in kb/wiki_linking_util.py,
            # as a method in WikiCandidateMentionGenerator
            # the output for get_mentions_with_gold is a dict with:
            # the input text tokenized, a list of candidate spans,
            # a list of candidate entities for each span,
            # priors for each candidate, and the gold entities for each span.
            processed = self.mention_generator.get_mentions_with_gold(" ".join(words), gold_spans,
                                                                      gold_entities, whitespace_tokenize=True, keep_gold_only=self.entity_disambiguation_only)
            if processed["candidate_spans"]:
                # this converts the above output into an allennlp Instance,
                # but for wiki_and_wordnet, it also generates extra candidates
                # by running the text through get_mentions_raw_text.
                # note that it uses a wordnet_mention_generator, not a
                # WikiCandidateMentionGenerator, like self.mention_generator
                # it is defined in kb/wordnet.py
                yield self.text_to_instance(doc_id=doc_id, **processed)

    def text_to_instance(self,
                         tokenized_text: List[str],
                         candidate_entities: List[List[str]],
                         candidate_spans: List[List[int]],
                         candidate_entity_prior: List[List[float]],
                         gold_entities: List[str] = None,
                         doc_id: str = None):

        assert doc_id is not None

        token_field = TextField([Token(x) for x in tokenized_text], self.token_indexers)

        # span fields cannot be instantiated with an empty list, like when there are no candidates.
        # create a dummy list of span fields, and grab a empty field to use.
        # https://github.com/allenai/allennlp/issues/1391
        # technically I don't need this anymore since even with no spans, the span list shouldn't be empty.
        if candidate_spans:
            span_fields = ListField([SpanField(*span, token_field) for span in candidate_spans])
        else:
            dummy = SpanField(0, 0, token_field)
            span_fields = ListField([dummy.empty_field()])

        candidate_entities = TextField(
                [Token(" ".join(candidate_list)) for candidate_list in candidate_entities],
                token_indexers=self.entity_indexer)

        max_cands = max(len(p) for p in candidate_entity_prior)
        for p in candidate_entity_prior:
            if len(p) < max_cands:
                p.extend([0.0] * (max_cands - len(p)))
        np_prior = np.array(candidate_entity_prior)
        prior_field = ArrayField(np_prior)

        # only one segment
        candidate_segment_ids = ArrayField(
                np.array([0] * len(candidate_entities)), dtype=np.int
        )

        fields = {
            "tokens": token_field,
            "candidate_spans": span_fields,
            "candidate_entities": candidate_entities,
            "candidate_entity_prior": prior_field,
            "candidate_segment_ids": candidate_segment_ids
            }
        if gold_entities:
            labels = TextField([Token(entity) for entity in gold_entities],
                               token_indexers=self.entity_indexer)
            fields["gold_entities"] = labels

        fields["doc_id"] = MetadataField(doc_id)

        # in Wiki, there is no extra candidate generator.
        # in W+W, there is one extra candidate generator.
        # wordnet_mention_generator defined in kb/wordnet.py
        if self.extra_candidate_generators:
            tokens = " ".join(tokenized_text)
            extra_candidates = {
                    key: generator.get_mentions_raw_text(tokens, whitespace_tokenize=True)
                    for key, generator in self.extra_candidate_generators.items()
            }
            fields['extra_candidates'] = MetadataField(extra_candidates)

        return Instance(fields, should_remap_span_indices=self.should_remap_span_indices)
