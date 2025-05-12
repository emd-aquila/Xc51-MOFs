import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .unimat import UniMatModel
from typing import Dict, Any, List


logger = logging.getLogger(__name__)

@register_model("unimof_ws24")
class UniMOFWS24Model(BaseUnicoreModel):
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--hidden-dim",
            type=int,
            default=128,
            help="output dimension of embedding",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            type=str,
            default="tanh",
            help="pooler activation function",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            default=0.1,
            help="pooler dropout",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.unimat = UniMatModel(self.args, dictionary)
        self.classifier = ClassificationHead(args.encoder_embed_dim, 
                                self.args.hidden_dim, 
                                self.args.num_classes, 
                                self.args.pooler_activation_fn,
                                self.args.pooler_dropout)
        self._cls_repr = None  # to store representation
        self.classifier.dense.register_forward_hook(self.save_cls_repr_hook)  # hook registration
    def save_cls_repr_hook(self, module, input, output):
        self._cls_repr = output

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        **kwargs
    ):
        """Forward pass for the UniMofWS24 model."""
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.unimat.gbf(dist, et)
            gbf_result = self.unimat.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
            
        padding_mask = src_tokens.eq(self.unimat.padding_idx)
        mol_x = self.unimat.embed_tokens(src_tokens)
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        encoder_outputs = self.unimat.encoder(mol_x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        cls_repr = encoder_outputs[0][:, 0, :] # CLS, shape of cls_repr is [batch_size, encoder_embed_dim]
        logits = self.classifier(cls_repr)

        return [logits]
       
    def extract_features(self, input):
        x = self.encoder(input)  # hypothetical encoder
        return x

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
@register_model("unimof_ws24_freeze")
class UniMOFWS24FreezeModel(BaseUnicoreModel):
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--hidden-dim",
            type=int,
            default=128,
            help="output dimension of embedding",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            type=str,
            default="tanh",
            help="pooler activation function",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            default=0.1,
            help="pooler dropout",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture_freeze(args)
        self.args = args
        self.unimat = UniMatModel(self.args, dictionary)

        # Freeze all parameters in the pretrained base model
        for param in self.unimat.parameters():
            param.requires_grad = False

        self.classifier = ClassificationHead(args.encoder_embed_dim, 
                                self.args.hidden_dim, 
                                self.args.num_classes, 
                                self.args.pooler_activation_fn,
                                self.args.pooler_dropout)
        self._cls_repr = None  # to store representation
        self.classifier.dense.register_forward_hook(self.save_cls_repr_hook)  # hook registration
    def save_cls_repr_hook(self, module, input, output):
        self._cls_repr = output
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        **kwargs
    ):
        """Forward pass for the UniMofWS24 model."""
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.unimat.gbf(dist, et)
            gbf_result = self.unimat.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
            
        padding_mask = src_tokens.eq(self.unimat.padding_idx)
        mol_x = self.unimat.embed_tokens(src_tokens)
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        encoder_outputs = self.unimat.encoder(mol_x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        cls_repr = encoder_outputs[0][:, 0, :] # CLS, shape of cls_repr is [batch_size, encoder_embed_dim]
        logits = self.classifier(cls_repr)

        return [logits]
    
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
@register_model("unimof_ws24_partial_freeze")
class UniMOFWS24PartialFreezeModel(BaseUnicoreModel):
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--hidden-dim",
            type=int,
            default=128,
            help="output dimension of embedding",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            type=str,
            default="tanh",
            help="pooler activation function",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            default=0.1,
            help="pooler dropout",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture_partial_freeze(args)
        self.args = args
        self.unimat = UniMatModel(self.args, dictionary)

        # Freeze all parameters in the pretrained base model
        for param in self.unimat.parameters():
            param.requires_grad = False

        # Unfreeze the last 4 layers of the encoder
        for name, param in self.unimat.named_parameters():
            if "encoder.layers.7" in name or "encoder.layers.6" in name or "encoder.layers.5" in name or "encoder.layers.4" in name:
                param.requires_grad = True

        self.classifier = ClassificationHead(args.encoder_embed_dim, 
                                self.args.hidden_dim, 
                                self.args.num_classes, 
                                self.args.pooler_activation_fn,
                                self.args.pooler_dropout)
        self._cls_repr = None  # to store representation
        self.classifier.dense.register_forward_hook(self.save_cls_repr_hook)  # hook registration
    def save_cls_repr_hook(self, module, input, output):
        self._cls_repr = output
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        **kwargs
    ):
        """Forward pass for the UniMofWS24 model."""
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.unimat.gbf(dist, et)
            gbf_result = self.unimat.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
            
        padding_mask = src_tokens.eq(self.unimat.padding_idx)
        mol_x = self.unimat.embed_tokens(src_tokens)
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        encoder_outputs = self.unimat.encoder(mol_x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        cls_repr = encoder_outputs[0][:, 0, :] # CLS, shape of cls_repr is [batch_size, encoder_embed_dim]
        logits = self.classifier(cls_repr)

        return [logits]
        
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
@register_model("unimof_ws24_mostly_freeze")
class UniMOFWS24MostlyFreezeModel(BaseUnicoreModel):
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--hidden-dim",
            type=int,
            default=128,
            help="output dimension of embedding",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            type=str,
            default="tanh",
            help="pooler activation function",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            default=0.1,
            help="pooler dropout",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture_mostly_freeze(args)
        self.args = args
        self.unimat = UniMatModel(self.args, dictionary)

        # Freeze all parameters in the pretrained base model
        for param in self.unimat.parameters():
            param.requires_grad = False

        # Unfreeze the last layer of the encoder
        for name, param in self.unimat.named_parameters():
            if "encoder.layers.7" in name:
                param.requires_grad = True

        self.classifier = ClassificationHead(args.encoder_embed_dim, 
                                self.args.hidden_dim, 
                                self.args.num_classes, 
                                self.args.pooler_activation_fn,
                                self.args.pooler_dropout)
        self._cls_repr = None  # to store representation
        self.classifier.dense.register_forward_hook(self.save_cls_repr_hook)  # hook registration
    def save_cls_repr_hook(self, module, input, output):
        self._cls_repr = output
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        **kwargs
    ):
        """Forward pass for the UniMofWS24 model."""
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.unimat.gbf(dist, et)
            gbf_result = self.unimat.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
            
        padding_mask = src_tokens.eq(self.unimat.padding_idx)
        mol_x = self.unimat.embed_tokens(src_tokens)
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        encoder_outputs = self.unimat.encoder(mol_x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        cls_repr = encoder_outputs[0][:, 0, :] # CLS, shape of cls_repr is [batch_size, encoder_embed_dim]
        logits = self.classifier(cls_repr)

        return [logits]
        
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("unimof_ws24", "unimof_ws24")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.lattice_loss = getattr(args, "lattice_loss", -1.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "relu")

@register_model_architecture("unimof_ws24_freeze", "unimof_ws24_freeze")
def base_architecture_freeze(args):
    base_architecture(args)

@register_model_architecture("unimof_ws24_partial_freeze", "unimof_ws24_partial_freeze")
def base_architecture_partial_freeze(args):
    base_architecture(args)

@register_model_architecture("unimof_ws24_mostly_freeze", "unimof_ws24_mostly_freeze")
def base_architecture_mostly_freeze(args):
    base_architecture(args)