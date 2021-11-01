from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.util import similarity, KgeLoss, rat

import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn
from torch.nn.parameter import Parameter

from pytorch_pretrained_bert.modeling import BertEncoder, BertConfig, BertLayerNorm, BertPreTrainedModel

from functools import partial

from kge.util import sc


class TrmEScorer(RelationalScorer):
    r"""Implementation of the TrmE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.dim = self.get_option("entity_embedder.dim")
        self.max_context_size = self.get_option("max_context_size")
        self.initializer_range = self.get_option("initializer_range")

        self.cls = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.cls, std=self.initializer_range)
        self.global_cls = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.global_cls, std=self.initializer_range)
        self.local_mask = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.local_mask, std=self.initializer_range)
        self.type_embeds = nn.Embedding(100, self.dim)
        torch.nn.init.normal_(self.type_embeds.weight, std=self.initializer_range)
        self.atomic_type_embeds = nn.Embedding(3, self.dim)
        torch.nn.init.normal_(self.atomic_type_embeds.weight, std=self.initializer_range)

        self.similarity = getattr(similarity, self.get_option("similarity"))(self.dim)
        self.layer_norm = BertLayerNorm(self.dim, eps=1e-12)
        self.atomic_layer_norm = BertLayerNorm(self.dim, eps=1e-12)

        self.transformer_encoder = rat.Encoder(
            lambda: rat.EncoderLayer(
                self.dim,
                rat.MultiHeadedAttentionWithRelations(
                    self.get_option("nhead"),
                    self.dim,
                    self.get_option("attn_dropout")),
                rat.PositionwiseFeedForward(
                    self.dim,
                    self.get_option("ff_dim"),
                    self.get_option("hidden_dropout")),
                num_relation_kinds=0,
                dropout=self.get_option("hidden_dropout")),
            self.get_option("nlayer"),
            self.initializer_range,
            tie_layers=False)

        config = BertConfig(0, hidden_size=self.dim,
                            num_hidden_layers=self.get_option("nlayer") // 2,
                            num_attention_heads=self.get_option("nhead"),
                            intermediate_size=self.get_option("ff_dim"),
                            hidden_act=self.get_option("activation"),
                            hidden_dropout_prob=self.get_option("hidden_dropout"),
                            attention_probs_dropout_prob=self.get_option("attn_dropout"),
                            max_position_embeddings=0,  # no effect
                            type_vocab_size=0,  # no effect
                            initializer_range=self.initializer_range)
        self.atom_encoder = BertEncoder(config)
        self.atom_encoder.config = config
        self.atom_encoder.apply(partial(BertPreTrainedModel.init_bert_weights, self.atom_encoder))

    def _get_encoder_output(self, e_emb, p_emb, ids, gt_ent, gt_rel, output_repr=False):
        n = p_emb.size(0)
        device = p_emb.device

        ctx_list, ctx_size = self.dataset.index('neighbor')
        ctx_ids = ctx_list[ids].to(device).transpose(1, 2)
        ctx_size = ctx_size[ids].to(device)

        # sample neighbors unifromly during training
        if self.training:
            perm_vector = sc.get_randperm_from_lengths(ctx_size, ctx_ids.size(1))
            ctx_ids = torch.gather(ctx_ids, 1, perm_vector.unsqueeze(-1).expand_as(ctx_ids))

        # [bs, length, 2]
        ctx_ids = ctx_ids[:, :self.max_context_size]
        ctx_size[ctx_size > self.max_context_size] = self.max_context_size

        # [bs, max_ctx_size]
        entity_ids = ctx_ids[...,0]
        relation_ids = ctx_ids[...,1]

        # initialize mask by length of context, seq=[CLS, S, N1, N2, ...]
        # 1 is the positions that will be attended to
        ctx_size = ctx_size + 2
        attention_mask = sc.get_mask_from_sequence_lengths(ctx_size, self.max_context_size + 2)

        if self.training and not output_repr:
            # mask out ground truth during training to avoid overfitting
            gt_mask = ((entity_ids != gt_ent.view(n, 1)) | (relation_ids != gt_rel.view(n, 1)))
            ctx_random_mask = (attention_mask
                               .new_ones((n, self.max_context_size))
                               .bernoulli_(1 - self.get_option("ctx_dropout")))
            attention_mask[:,2:] = attention_mask[:, 2:] & ctx_random_mask & gt_mask

        # [bs, max_ctx_size, dim]
        entity_emb = self._entity_embedder().embed(entity_ids)
        relation_emb = self._relation_embedder().embed(relation_ids)

        if self.training and self.get_option("self_dropout") > 0 and self.max_context_size > 0 and not output_repr:
            # sample a proportion of input for masked prediction similar to the MLM in BERT
            self_dropout_sample = sc.get_bernoulli_mask([n], self.get_option("self_dropout"), device)

            # replace with mask tokens
            masked_sample = sc.get_bernoulli_mask([n], self.get_option("mlm_mask"), device) & self_dropout_sample
            e_emb[masked_sample] = self.local_mask.unsqueeze(0)

            # replace with random sampled entities, no back propagation here
            replaced_sample = sc.get_bernoulli_mask([n], self.get_option(
                "mlm_replace"), device) & self_dropout_sample & ~masked_sample
            e_emb[replaced_sample] = self._entity_embedder().embed(torch.randint(self.dataset.num_entities(
            ), replaced_sample.shape, dtype=torch.long, device=device))[replaced_sample].detach()

        src = torch.cat([torch.stack([e_emb, p_emb], dim=1), torch.stack([entity_emb, relation_emb], dim=2)
                         .view(n, 2 * self.max_context_size, self.dim)], dim=1)
        src = src.reshape(n, self.max_context_size + 1, 2, self.dim)

        # only keep un-masked positions to reduce computational cost
        src = src[attention_mask[:, 1:]]

        # add CLS (local) and pos embedding
        pos = self.atomic_type_embeds(torch.arange(0, 3, device=device)).unsqueeze(0).repeat(src.shape[0], 1, 1)
        src = torch.cat([self.cls.expand(src.size(0), 1, self.dim), src], dim=1) + pos

        src = F.dropout(src, p=self.get_option("output_dropout"), training=self.training and not output_repr)
        src = self.atomic_layer_norm(src)

        # [bs, dim]
        out = self.atom_encoder(src,
                                self.convert_mask(src.new_ones(src.size(0), src.size(1), dtype=torch.long)),
                                output_all_encoded_layers=False)[-1][:,0]

        # recover results from output based on mask
        src = out.new_zeros(n, self.max_context_size + 1, self.dim)
        src[attention_mask[:, 1:]] = out

        # when not using graph context, exit here
        if self.max_context_size == 0:
            return src[:, 0], 0

        # begin the processing of graph context with the upper transformer block
        # add CLS (global) and pos embeddings
        src = torch.cat([self.global_cls.expand(n, 1, self.dim), src], dim=1)
        pos = self.type_embeds(torch.arange(0, 3, device=device))
        src[:, 0] = src[:, 0] + pos[0].unsqueeze(0)
        src[:, 1] = src[:, 1] + pos[1].unsqueeze(0)
        src[:, 2:] = src[:, 2:] + pos[2].view(1, 1, -1)

        src = F.dropout(src, p=self.get_option("hidden_dropout"), training=self.training)
        src = self.layer_norm(src)
        out = self.transformer_encoder(src, None, self.convert_mask_rat(attention_mask))

        if output_repr:
            return out, self.convert_mask(attention_mask)

        out = out[-1][:,:2]

        # compute the mlm-like loss if needed
        if self.training and self.get_option("add_mlm_loss") and self.get_option("self_dropout") > 0.0 and self_dropout_sample.sum() > 0:
            all_entity_emb = self._entity_embedder().embed_all()
            all_entity_emb = F.dropout(all_entity_emb, p=self.get_option("output_dropout"), training=self.training)
            source_scores = self.similarity(out[:, 1], all_entity_emb, False).view(n, -1)
            self_pred_loss = F.cross_entropy(
                source_scores[self_dropout_sample], ids[self_dropout_sample], reduction='mean')
            return out[:, 0], self_pred_loss
        else:
            return out[:, 0], 0

    def convert_mask_rat(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
        return attention_mask

    def convert_mask(self, attention_mask):
        # extend mask to Transformer format
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0
        return attention_mask

    def _scoring(self, s_emb, p_emb, o_emb, is_pairwise, ids, gt_ent, gt_rel):
        encoder_output, self_pred_loss = self._get_encoder_output(s_emb, p_emb, ids, gt_ent, gt_rel)
        o_emb = F.dropout(o_emb, p=self.get_option("output_dropout"), training=self.training)
        target_scores = self.similarity(encoder_output, o_emb, is_pairwise).view(p_emb.size(0), -1)
        if self.training:
            return target_scores, self_pred_loss
        else:
            return target_scores

    def score_emb(self, s_emb, p_emb, o_emb, combine: str, s, o, gt_ent=None, gt_rel=None):
        # the 'spo' combination is only used in reciprocal model
        # 'spo' prefix stands for triple-wise scoring, suffix "s"/"o" indicates direction
        if combine == 'spoo' or combine == 'sp_' or combine == 'spo':
            out = self._scoring(s_emb, p_emb, o_emb, combine.startswith('spo'), s, gt_ent, gt_rel)
        elif combine == 'spos' or combine == '_po':
            out = self._scoring(o_emb, p_emb, s_emb, combine.startswith('spo'), o, gt_ent, gt_rel)
        else:
            raise Exception("Combine {} is not supported in TrmE's score function".format(combine))
        return out


class TrmE(KgeModel):
    r"""Implementation of the TrmE KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(
            config, dataset, TrmEScorer, configuration_key=configuration_key
        )
        self.loss = KgeLoss.create(config)

    def forward(self, fn_name, *args, **kwargs):
        # bind entity/relation embedder to scorer to retrieve embeddings
        self._scorer._entity_embedder = self.get_s_embedder
        self._scorer._relation_embedder = self.get_p_embedder

        # call score_sp/score_po during training, score_spo/score_sp_po during inference
        scores = getattr(self, fn_name)(*args, **kwargs)

        # delete references to embedder getter
        del self._scorer._entity_embedder
        del self._scorer._relation_embedder

        if fn_name == 'get_hitter_repr':
            return scores

        if self.training:
            self_loss_w = self.get_option("self_dropout")
            # MLM-like loss is weighted by the proportion of entities sampled
            self_loss_w = self_loss_w / (1 + self_loss_w)
            return self.loss(scores[0], kwargs["gt_ent"]) + self_loss_w * scores[1] * scores[0].size(0)
        else:
            return scores

    def get_hitter_repr(self, s, p):
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        return self._scorer._get_encoder_output(s_emb, p_emb, s, None, None, output_repr=True)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        if direction:
            if direction == 's':
                p_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
            else:
                p_emb = self.get_p_embedder().embed(p)
            return self._scorer.score_emb(s_emb, p_emb, o_emb, "spo" + direction, s, o).view(-1)
        else:
            raise Exception(
                "The TrmE model cannot compute "
                "undirected spo scores."
            )

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None, gt_ent=None, gt_rel=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        if o is None:
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)

        return self._scorer.score_emb(s_emb, p_emb, o_emb, "sp_", s, None, gt_ent, gt_rel)

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None, gt_ent=None, gt_rel=None) -> Tensor:
        if s is None:
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        p_inv_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())

        return self._scorer.score_emb(s_emb, p_inv_emb, o_emb, "_po", None, o, gt_ent, gt_rel)

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        p_inv_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
        o_emb = self.get_o_embedder().embed(o)
        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(
                s_emb, p_emb, all_entities, "sp_", s, None)
            po_scores = self._scorer.score_emb(
                all_entities, p_inv_emb, o_emb, "_po", None, o)
        else:
            assert False
        return torch.cat((sp_scores, po_scores), dim=1)
