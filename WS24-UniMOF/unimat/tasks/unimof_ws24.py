import logging
import os
import torch

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    RawArrayDataset,
    RawNumpyDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
)
from unimat.data import (
    KeyDataset,
    LMDBDataset,
    ToTorchDataset,
    MaskPointsDataset,
    DistanceDataset,
    EdgeTypeDataset,
    PrependAndAppend2DDataset,
    RightPadDatasetCoord,
    LatticeNormalizeDataset,
    RemoveHydrogenDataset,
    CroppingDataset,
    NormalizeDataset,
    NumericalTransformDataset,
)
from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils

logger = logging.getLogger(__name__)

@register_task("unimof_ws24")
class UniMOFWS24Task(UnicoreTask):
    """Task for training transformer auto-encoder model on WS24 dataset"""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path"
        )
        parser.add_argument(
            "--task-name",
            type=str,
            default='',
            help="downstream task name"
        )
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name"
        )
        parser.add_argument(
            "--num-classes",
            default=4,
            type=int,
            help="finetune downstream task classes numbers"
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=512,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--finetune-mol-model",
            help="load unimat finetune model",
        )
        parser.add_argument(
            "--weight-by-class",
            help="weight loss by class counts",
            action="store_true",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.class_counts = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        tgt_dataset = KeyDataset(dataset, "target")
        tgt_dataset = ToTorchDataset(tgt_dataset, dtype='float32')
        name_dataset = KeyDataset(dataset, "mof-name")
        #name_dataset = ToTorchDataset(tgt_dataset, dtype='string')
        if self.args.remove_hydrogen:
            dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates")
        dataset = CroppingDataset(dataset, self.seed, "atoms", "coordinates", self.args.max_atoms)
        dataset = NormalizeDataset(dataset, "coordinates")
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        coord_dataset = KeyDataset(dataset, "coordinates")

        #mof_name = KeyDataset(dataset, "mof-name")

        if self.args.weight_by_class:
            all_targets = []
            for i in range(len(tgt_dataset)):
                try:
                    target = tgt_dataset[i]
                    if target is not None:
                        all_targets.append(int(target))
                    else:
                        logger.warning(f"Sample {i} in tgt_dataset is None.")
                except Exception as e:
                    logger.error(f"Failed to load sample {i}: {e}")

            if not all_targets:
                raise ValueError("No valid targets found in tgt_dataset.")

            all_targets = torch.tensor(all_targets)
            all_targets -= 1  # Make labels zero-indexed
            self.class_counts = torch.bincount(all_targets, minlength=self.args.num_classes)
            logger.info(f"Class counts: {self.class_counts}")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = ToTorchDataset(coord_dataset, 'float32')
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        nest_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.dictionary.pad(),
                        ),
                        "src_coord": RightPadDatasetCoord(
                            coord_dataset,
                            pad_idx=0,
                        ),
                        "src_distance": RightPadDataset2D(
                            distance_dataset,
                            pad_idx=0,
                        ),
                        "src_edge_type": RightPadDataset2D(
                            edge_type,
                            pad_idx=0,
                        ),
                    },
                    "mof_name": name_dataset,
                    "target": {
                        "finetune_target": tgt_dataset,
                    },
                },
            )
        if split in ["train", "train.small"]:
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models  
        model = models.build_model(args, self)
        if args.finetune_mol_model is not None:
                print("load pretrain model weight from...", args.finetune_mol_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_mol_model,
                )
                model.unimat.load_state_dict(state["model"], strict=False)
        return model