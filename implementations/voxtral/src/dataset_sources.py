"""Resolve Hugging Face snapshots and iterate local Parquet without I/O threads."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def resolve_dataset_snapshot(dataset_id: str) -> Path:
    """Return a local snapshot, downloading it to the active tmp cache if absent."""
    try:
        snapshot = snapshot_download(
            dataset_id,
            repo_type="dataset",
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        snapshot = snapshot_download(
            dataset_id,
            repo_type="dataset",
            local_files_only=False,
        )
    return Path(snapshot)


def _parquet_files(snapshot: Path, config: str, split: str) -> tuple[Path, ...]:
    directory_layout = tuple(sorted((snapshot / config / split).glob("*.parquet")))
    if directory_layout:
        return directory_layout

    flat_layout = tuple(sorted((snapshot / config).glob(f"{split}-*.parquet")))
    if flat_layout:
        return flat_layout

    raise FileNotFoundError(
        f"No Parquet files for config={config!r}, split={split!r} "
        f"under snapshot {snapshot}"
    )


@dataclass(frozen=True)
class LocalParquetDataset:
    files: tuple[Path, ...]
    shuffle_seed: int | None = None
    shuffle_buffer_size: int = 10_000
    read_batch_size: int = 64
    selected_indices: tuple[int, ...] | None = None

    def _all_rows(self, columns=None):
        for path in self.files:
            parquet = pq.ParquetFile(path, memory_map=True)
            try:
                for batch in parquet.iter_batches(
                    batch_size=self.read_batch_size,
                    columns=columns,
                    use_threads=False,
                ):
                    yield from batch.to_pylist()
            finally:
                del parquet

    def _selected_rows(self, columns=None):
        targets = self.selected_indices or ()
        target_offset = 0
        global_start = 0
        for path in self.files:
            parquet = pq.ParquetFile(path, memory_map=True)
            try:
                for row_group in range(parquet.metadata.num_row_groups):
                    row_count = parquet.metadata.row_group(row_group).num_rows
                    global_end = global_start + row_count
                    local_offsets = []
                    while (
                        target_offset < len(targets)
                        and targets[target_offset] < global_end
                    ):
                        index = targets[target_offset]
                        if index < global_start:
                            raise ValueError(
                                "Selected dataset indices must be sorted and unique."
                            )
                        local_offsets.append(index - global_start)
                        target_offset += 1
                    if local_offsets:
                        table = parquet.read_row_group(
                            row_group,
                            columns=columns,
                            use_threads=False,
                        )
                        selected = table.take(pa.array(local_offsets, type=pa.int64()))
                        yield from selected.to_pylist()
                    global_start = global_end
            finally:
                del parquet

        if target_offset != len(targets):
            raise IndexError(
                f"Selected row index {targets[target_offset]} exceeds dataset size "
                f"{global_start}."
            )

    def _rows(self, columns=None):
        if self.selected_indices is None:
            yield from self._all_rows(columns=columns)
        else:
            yield from self._selected_rows(columns=columns)

    def __iter__(self):
        rows = self._rows()
        if self.shuffle_seed is None:
            yield from rows
            return

        rng = random.Random(self.shuffle_seed)
        buffer = []
        for row in rows:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(row)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row

        rng.shuffle(buffer)
        yield from buffer

    def shuffle(self, seed: int, buffer_size: int = 10_000):
        if self.selected_indices is not None:
            raise ValueError("Cannot shuffle a dataset after selecting fixed row indices.")
        return replace(
            self,
            shuffle_seed=int(seed),
            shuffle_buffer_size=int(buffer_size),
        )

    def select_indices(self, indices):
        selected = tuple(int(index) for index in indices)
        if selected != tuple(sorted(set(selected))):
            raise ValueError("Selected dataset indices must be sorted and unique.")
        return replace(
            self,
            selected_indices=selected,
            shuffle_seed=None,
        )

    def iter_columns(self, columns):
        """Iterate selected columns without decoding unrelated audio payloads."""
        yield from self._rows(columns=list(columns))

    @property
    def num_rows(self) -> int:
        return sum(
            pq.ParquetFile(path, memory_map=True).metadata.num_rows
            for path in self.files
        )


def load_local_parquet_dataset(
    dataset_id: str,
    config: str,
    split: str,
) -> LocalParquetDataset:
    """Resolve one config/split to a node-local, single-threaded row iterator."""
    snapshot = resolve_dataset_snapshot(dataset_id)
    return LocalParquetDataset(_parquet_files(snapshot, config, split))
