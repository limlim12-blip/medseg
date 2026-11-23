import torch
from torch.utils.data import DataLoader

from .dataset import build_datasets


class MedicalDataLoader(DataLoader):
    """
    Training DataLoader over a regular (map-style) dataset.

    Same interface as dluet's ``IDataLoader``: the object itself iterates the
    training split, ``split_validation()`` returns the validation loader, and
    ``split_test()`` returns the test loader.
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers=0,
        img_size=256,
        validation_split=0.1,
        test_split=0.1,
        seed=123,
        collate_fn=None,
        download=True,
    ):
        img_size = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        train_ds, valid_ds, test_ds = build_datasets(
            data_dir, img_size, validation_split, test_split, seed, download
        )
        self.n_samples = len(train_ds)
        self.img_size = img_size

        pin = torch.cuda.is_available()
        persistent = num_workers > 0
        generator = torch.Generator()
        generator.manual_seed(seed)

        super().__init__(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=True,  # possible because the dataset is map-style
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin,
            persistent_workers=persistent,
            generator=generator,
        )

        eval_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin,
            persistent_workers=persistent,
        )
        self.valid_loader = (
            DataLoader(valid_ds, **eval_kwargs) if len(valid_ds) else None
        )
        self.test_loader = DataLoader(test_ds, **eval_kwargs) if len(test_ds) else None

    def split_validation(self):
        return self.valid_loader

    def split_test(self):
        return self.test_loader
