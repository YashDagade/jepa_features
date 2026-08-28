"""Reusable frozen-feature location probes for Tunnel Rooms diagnostics.

The location probe is deliberately a diagnostic, not a training objective. A
fixed sidecar selects episode-disjoint samples and stores privileged coordinate
labels outside the training HDF5. Separate current and predicted visual vectors
measure where each location is linearly available without cross-feature leakage.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_READOUT = 'separate_current_predicted_visual'
FEATURE_SOURCES = ('current_visual', 'predicted_visual')
FEATURE_CACHE_SCHEMA = 3
SIDECAR_SCHEMAS = (1, 2)
SPLIT_NAMES = ('fit', 'validation', 'test')
SPLIT_TO_CODE = {name: code for code, name in enumerate(SPLIT_NAMES)}


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in '0123456789abcdef' for character in value
    )


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Hash a source artifact without loading it into memory."""
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _target_names(h5: h5py.File) -> tuple[str, ...]:
    if 'target_names' in h5:
        return tuple(_decode(value) for value in h5['target_names'][:])
    encoded = h5.attrs.get('target_names_json')
    if encoded is None:
        raise ValueError('Location sidecar is missing target_names.')
    return tuple(str(value) for value in json.loads(_decode(encoded)))


def _split_codes(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        codes = values.astype(np.int8, copy=False)
    else:
        aliases = {'val': 'validation'}
        names = [
            aliases.get(_decode(value), _decode(value)) for value in values
        ]
        try:
            codes = np.asarray(
                [SPLIT_TO_CODE[name] for name in names], np.int8
            )
        except KeyError as error:
            raise ValueError(
                f'Unknown location-probe split {error.args[0]!r}.'
            ) from error
    if not np.isin(codes, np.arange(len(SPLIT_NAMES))).all():
        raise ValueError(
            f'Location-probe split codes must be 0, 1, or 2: {codes}.'
        )
    return codes


def _hash_array(hasher: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    hasher.update(name.encode('utf-8'))
    hasher.update(array.dtype.str.encode('ascii'))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes())


def _finite_or_zero(value: torch.Tensor) -> torch.Tensor:
    """Replace reset-boundary NaNs without changing valid model inputs."""
    return torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _frame_feature(value: torch.Tensor) -> torch.Tensor:
    """Return the one global CLS vector selected for a single frame."""
    # 1. These experiments use the published global-token model contract.
    if value.ndim != 3 or value.size(1) != 1:
        raise ValueError(
            'Location features require exactly one selected time step, got '
            f'{tuple(value.shape)}.'
        )
    # 2. Never pool or infer an alternative representation here.
    return value[:, 0]


def sample_sha256(
    sample_ids: np.ndarray,
    episode_ids: np.ndarray,
    frame_rows: np.ndarray,
    future_rows: np.ndarray,
    splits: np.ndarray,
    targets: np.ndarray,
    target_names: Sequence[str],
) -> str:
    """Hash every field that defines a fixed location-probe sample set."""
    hasher = hashlib.sha256()
    for name, value in (
        ('sample_id', sample_ids),
        ('episode_id', episode_ids),
        ('frame_rows', frame_rows),
        ('future_row', future_rows),
        ('split', splits),
        ('target', targets),
    ):
        _hash_array(hasher, name, np.asarray(value))
    hasher.update(
        json.dumps(list(target_names), separators=(',', ':')).encode()
    )
    return hasher.hexdigest()


def assert_episode_disjoint(
    episode_ids: np.ndarray, split_codes: np.ndarray
) -> None:
    """Reject leakage when any episode appears in more than one split."""
    episodes = np.asarray(episode_ids, dtype=np.int64)
    splits = _split_codes(split_codes)
    seen: dict[int, int] = {}
    for episode, split in zip(episodes, splits, strict=True):
        previous = seen.setdefault(int(episode), int(split))
        if previous != int(split):
            raise ValueError(
                f'Episode {episode} appears in both '
                f'{SPLIT_NAMES[previous]} and {SPLIT_NAMES[int(split)]}.'
            )


def _coordinate_columns(names: Sequence[str], prefix: str) -> tuple[int, int]:
    expected = (f'{prefix}_x', f'{prefix}_y')
    try:
        return tuple(names.index(name) for name in expected)  # type: ignore[return-value]
    except ValueError as error:
        raise ValueError(
            f'Location targets must include {expected[0]!r} and {expected[1]!r}.'
        ) from error


def _validate_target_names(names: Sequence[str]) -> None:
    """Require current/future agent and complete endpoint coordinate pairs."""
    if len(set(names)) != len(names):
        raise ValueError('Location target names must be unique.')
    agent_names = {
        'agent_current_x',
        'agent_current_y',
        'agent_future_x',
        'agent_future_y',
    }
    _coordinate_columns(names, 'agent_current')
    _coordinate_columns(names, 'agent_future')
    endpoint_names = [name for name in names if name not in agent_names]
    if not endpoint_names:
        raise ValueError('Location targets must include tunnel endpoints.')
    prefixes = {
        name[:-2] for name in endpoint_names if name.endswith(('_x', '_y'))
    }
    if len(prefixes) * 2 != len(endpoint_names):
        raise ValueError(
            'Every tunnel endpoint requires exactly one x/y target pair.'
        )
    for prefix in prefixes:
        _coordinate_columns(names, prefix)


def hdf5_column_stats(
    dataset: h5py.Dataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the finite-row z-score statistics used by WM training loaders."""
    values = dataset[:].reshape(-1, dataset.shape[-1])
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) == 0:
        raise ValueError(
            f'{dataset.name} has no finite rows for normalization.'
        )
    mean = values.mean(axis=0).astype(np.float32)
    std = np.maximum(values.std(axis=0), 1e-8).astype(np.float32)
    return torch.from_numpy(mean), torch.from_numpy(std)


def aligned_history_inputs(
    source: h5py.File,
    frame_rows: np.ndarray,
    frameskip: int,
    *,
    sample_id: int,
) -> dict[str, torch.Tensor]:
    """Read one frameskip-aligned visual/action history for a frozen probe."""
    rows = np.asarray(frame_rows, dtype=np.int64)

    # 1. Assemble the same primitive-action chunks used by WM training.
    chunks = []
    for row in rows:
        action = source['action'][row : row + frameskip]
        if len(action) != frameskip:
            raise IndexError(f'Incomplete action chunk at source row {row}.')
        chunks.append(np.asarray(action, dtype=np.float32).reshape(-1))

    # 2. Fail before inference if any recorded action is invalid.
    action_history = np.stack(chunks)
    if not np.isfinite(action_history).all():
        raise FloatingPointError(
            f'Non-finite action in frozen-probe sample {sample_id}.'
        )

    # 3. Return model inputs only; each probe owns its privileged labels.
    item = {
        'pixels': torch.from_numpy(source['pixels'][rows]),
        'action_history': torch.from_numpy(action_history),
        'sample_id': torch.tensor(sample_id, dtype=torch.int64),
    }
    return item


@dataclass(frozen=True)
class SidecarMetadata:
    """Immutable metadata needed to interpret one location-probe sidecar."""

    source_hdf5: str
    source_sha256: str
    sample_sha256: str
    frameskip: int
    num_preds: int
    history_size: int
    target_names: tuple[str, ...]


class LocationProbeSidecarDataset(Dataset):
    """Resolve fixed sidecar rows against a read-only training HDF5.

    Sidecar schemas ``1`` and ``2`` contain ``sample_id``, ``episode_id``,
    ``split``, ``frame_rows``, ``future_row``, and ``target`` datasets.
    ``frame_rows`` are absolute source-HDF5 rows sampled at ``frameskip``
    cadence. The final row is the current frame; ``future_row`` is exactly
    ``num_preds * frameskip`` rows later. Coordinate names live in
    ``target_names`` or ``target_names_json``. The source path and hashes are
    HDF5 attributes.
    """

    def __init__(
        self,
        sidecar_path: str | Path,
        source_hdf5: str | Path | None = None,
        *,
        validate: bool = True,
    ) -> None:
        # 1. Load the immutable sample indices, labels, and provenance metadata.
        self.sidecar_path = Path(sidecar_path)
        self._source: h5py.File | None = None
        with h5py.File(self.sidecar_path, 'r') as h5:
            schema = int(h5.attrs.get('schema_version', -1))
            if schema not in SIDECAR_SCHEMAS:
                raise ValueError(
                    f'Unsupported location sidecar schema {schema}; '
                    f'expected one of {SIDECAR_SCHEMAS}.'
                )
            required = (
                'sample_id',
                'episode_id',
                'split',
                'frame_rows',
                'future_row',
                'target',
            )
            missing = [name for name in required if name not in h5]
            if missing:
                raise ValueError(
                    f'Location sidecar is missing datasets: {missing}.'
                )
            self.sample_ids = h5['sample_id'][:].astype(np.int64)
            self.episode_ids = h5['episode_id'][:].astype(np.int64)
            self.split_codes = _split_codes(h5['split'][:])
            self.frame_rows = h5['frame_rows'][:].astype(np.int64)
            self.future_rows = h5['future_row'][:].astype(np.int64)
            self.targets = h5['target'][:].astype(np.float32)
            names = _target_names(h5)
            stored_source = _decode(h5.attrs.get('source_hdf5', ''))
            stored_hash = _decode(h5.attrs.get('source_sha256', ''))
            stored_sample_hash = _decode(h5.attrs.get('sample_sha256', ''))
            frameskip = int(h5.attrs.get('frameskip', 0))
            num_preds = int(h5.attrs.get('num_preds', 0))
            history_size = int(
                h5.attrs.get('history_size', self.frame_rows.shape[-1])
            )

        # 2. Resolve the source path without allowing a caller to change corpora.
        if not stored_source:
            raise ValueError('Location sidecar has no source_hdf5 attribute.')
        recorded_source = Path(stored_source)
        if not recorded_source.is_absolute():
            recorded_source = self.sidecar_path.parent / recorded_source
        self.source_path = (
            recorded_source if source_hdf5 is None else Path(source_hdf5)
        )
        if self.source_path.resolve() != recorded_source.resolve():
            raise ValueError(
                'Configured source HDF5 does not match the sidecar source: '
                f'{self.source_path} != {recorded_source}.'
            )
        # 3. Materialize one compatibility contract and validate it eagerly.
        self.metadata = SidecarMetadata(
            source_hdf5=str(self.source_path),
            source_sha256=stored_hash,
            sample_sha256=stored_sample_hash,
            frameskip=frameskip,
            num_preds=num_preds,
            history_size=history_size,
            target_names=names,
        )
        self._validate_shapes()
        if validate:
            self.validate()
        else:
            self._validate_source_hash()
        # 4. Reuse the training corpus statistics for probe-model inputs.
        source = self._open_source()
        self.action_input_mean, self.action_input_std = hdf5_column_stats(
            source['action']
        )

    def _validate_shapes(self) -> None:
        count = len(self.sample_ids)
        _validate_target_names(self.metadata.target_names)
        one_dimensional = (
            self.episode_ids,
            self.split_codes,
            self.future_rows,
        )
        if any(value.shape != (count,) for value in one_dimensional):
            raise ValueError(
                'Sidecar IDs, splits, and future rows must be one-dimensional.'
            )
        if self.frame_rows.shape != (count, self.metadata.history_size):
            raise ValueError('frame_rows does not match count x history_size.')
        if self.targets.shape != (count, len(self.metadata.target_names)):
            raise ValueError('target does not match count x target_names.')
        if count == 0 or len(np.unique(self.sample_ids)) != count:
            raise ValueError(
                'Location sidecar sample IDs must be non-empty and unique.'
            )
        if not np.isfinite(self.targets).all():
            raise ValueError(
                'Location sidecar targets contain NaN or infinity.'
            )
        if self.metadata.frameskip <= 0:
            raise ValueError('frameskip must be positive.')
        if self.metadata.num_preds != 1:
            raise ValueError(
                'Location probes require num_preds == 1 because the current '
                'predictor action contract supplies one future action chunk.'
            )
        if not _is_sha256(self.metadata.source_sha256) or not _is_sha256(
            self.metadata.sample_sha256
        ):
            raise ValueError(
                'Location sidecar must record lowercase source and sample '
                'SHA-256 hashes.'
            )

    def _validate_source_hash(self) -> None:
        if not self.source_path.is_file():
            raise FileNotFoundError(
                f'Location-probe source does not exist: {self.source_path}'
            )
        actual_hash = _sha256_file(self.source_path)
        if actual_hash != self.metadata.source_sha256:
            raise ValueError(
                'Location sidecar source HDF5 SHA-256 does not match the '
                f'source contents: {actual_hash} != '
                f'{self.metadata.source_sha256}.'
            )

    def _open_source(self) -> h5py.File:
        if self._source is None:
            self._source = h5py.File(self.source_path, 'r', swmr=True)
        return self._source

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state['_source'] = None
        return state

    @property
    def action_chunk_dim(self) -> int:
        source = self._open_source()
        return self.metadata.frameskip * int(
            np.prod(source['action'].shape[1:])
        )

    def validate(self) -> None:
        """Validate hashes, temporal alignment, labels, and episode isolation."""
        # 1. Establish that the sidecar belongs to this corpus and has no split leakage.
        self._validate_source_hash()
        assert_episode_disjoint(self.episode_ids, self.split_codes)

        # 2. Verify that every history and future target follows training cadence.
        cadence = np.diff(self.frame_rows, axis=1)
        if cadence.size and not np.all(cadence == self.metadata.frameskip):
            raise ValueError(
                'Sidecar frame rows are not separated by frameskip.'
            )
        expected_future = self.frame_rows[:, -1] + (
            self.metadata.num_preds * self.metadata.frameskip
        )
        if not np.array_equal(self.future_rows, expected_future):
            raise ValueError(
                'future_row is not aligned by num_preds * frameskip.'
            )
        # 3. Recompute the sample-set hash before trusting any stored labels.
        actual_hash = sample_sha256(
            self.sample_ids,
            self.episode_ids,
            self.frame_rows,
            self.future_rows,
            self.split_codes,
            self.targets,
            self.metadata.target_names,
        )
        if actual_hash != self.metadata.sample_sha256:
            raise ValueError(
                'Location sidecar sample SHA-256 does not match its contents.'
            )

        # 4. Check each referenced row, episode, and privileged agent label.
        current_columns = _coordinate_columns(
            self.metadata.target_names, 'agent_current'
        )
        future_columns = _coordinate_columns(
            self.metadata.target_names, 'agent_future'
        )
        source = self._open_source()
        if 'pos_agent' not in source or 'ep_idx' not in source:
            raise ValueError('Source HDF5 must contain pos_agent and ep_idx.')
        row_count = len(source['pos_agent'])
        all_rows = np.concatenate(
            (self.frame_rows.reshape(-1), self.future_rows)
        )
        if np.any(all_rows < 0) or np.any(all_rows >= row_count):
            raise ValueError(
                'Sidecar references rows outside the source HDF5.'
            )
        for index in range(len(self)):
            rows = self.frame_rows[index]
            episode = self.episode_ids[index]
            checked_rows = np.concatenate(
                (rows, self.future_rows[index : index + 1])
            )
            if not np.all(source['ep_idx'][checked_rows] == episode):
                raise ValueError(
                    f'Sample {self.sample_ids[index]} crosses episode boundaries.'
                )
            current = source['pos_agent'][rows[-1]]
            future = source['pos_agent'][self.future_rows[index]]
            if not np.array_equal(
                self.targets[index, current_columns], current
            ):
                raise ValueError(
                    f'Sample {self.sample_ids[index]} has a misaligned current label.'
                )
            if not np.array_equal(self.targets[index, future_columns], future):
                raise ValueError(
                    f'Sample {self.sample_ids[index]} has a misaligned future label.'
                )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source = self._open_source()
        rows = self.frame_rows[index]

        # 1. Reuse the shared aligned-history reader, then attach this probe's
        # immutable coordinate target and episode-disjoint split.
        item = aligned_history_inputs(
            source,
            rows,
            self.metadata.frameskip,
            sample_id=int(self.sample_ids[index]),
        )
        item['target'] = torch.from_numpy(self.targets[index])
        item['split'] = torch.tensor(self.split_codes[index], dtype=torch.int8)
        return item


class FrozenLocationFeatures(nn.Module):
    """Extract current and predicted visual features from a frozen world model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        image_size: int = 224,
        history_size: int = 3,
        action_input_mean: torch.Tensor | None = None,
        action_input_std: torch.Tensor | None = None,
        freeze_model: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = int(image_size)
        self.history_size = int(history_size)
        self._feature_width: int | None = None
        if not all(
            hasattr(model, name)
            for name in ('encoder', 'projector', 'action_encoder', 'predict')
        ):
            raise ValueError(
                'Frozen probes require a global-CLS LeWM-family model.'
            )
        self.register_buffer(
            'mean', torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            'action_input_mean',
            torch.as_tensor(
                action_input_mean
                if action_input_mean is not None
                else torch.zeros(1)
            ).float(),
        )
        self.register_buffer(
            'action_input_std',
            torch.as_tensor(
                action_input_std
                if action_input_std is not None
                else torch.ones(1)
            )
            .float()
            .clamp_min(1e-8),
        )
        if freeze_model:
            self.model.eval()
            self.model.requires_grad_(False)

    def prepare_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Convert uint8 NHWC histories to normalized NTCWH tensors."""
        pixels = pixels.to(self.mean.device)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        if pixels.shape[-1] in (1, 3, 4):
            pixels = pixels[..., :3].permute(0, 1, 4, 2, 3)
        pixels = pixels.float()
        if pixels.max() > 2.0:
            pixels = pixels / 255.0
        batch, steps = pixels.shape[:2]
        flat = rearrange(pixels, 'b t c h w -> (b t) c h w')
        flat = F.interpolate(
            flat,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )
        pixels = rearrange(flat, '(b t) c h w -> b t c h w', b=batch, t=steps)
        return (pixels - self.mean) / self.std

    def normalize_action_history(self, action: torch.Tensor) -> torch.Tensor:
        """Match the primitive-action normalization used during WM training."""
        primitive_dim = int(self.action_input_mean.numel())
        if primitive_dim <= 1:
            return action
        shape = action.shape
        action = action.view(*shape[:2], -1, primitive_dim)
        mean = self.action_input_mean.view(1, 1, 1, primitive_dim)
        std = self.action_input_std.view(1, 1, 1, primitive_dim)
        return ((action - mean) / std).view(shape)

    def _encode_history(
        self,
        pixels: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return projected global-CLS history and its prediction."""
        info = {'pixels': pixels, 'action': action}
        info = self.model.encode(info)
        prediction = self.model.predict(info['emb'], info['act_emb'])
        return info['emb'], prediction, int(info['emb'].shape[-1])

    def _encode_stage_history(
        self,
        pixels: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return actual encoder CLS, projector, and predictor histories."""
        # 1. Read the actual CLS token emitted by the published encoder.
        batch, steps = pixels.shape[:2]
        encoder_dtype = next(self.model.encoder.parameters()).dtype
        flat_pixels = rearrange(
            pixels.to(encoder_dtype), 'b t c h w -> (b t) c h w'
        )
        output = self.model.encoder(flat_pixels, interpolate_pos_encoding=True)
        encoder = rearrange(
            output.last_hidden_state[:, 0],
            '(b t) d -> b t d',
            b=batch,
            t=steps,
        )

        # 2. Apply the model's published projector and action encoder exactly once.
        projected = self.model.projector(
            rearrange(encoder, 'b t d -> (b t) d')
        )
        projected = rearrange(projected, '(b t) d -> b t d', b=batch, t=steps)
        action_embedding = self.model.action_encoder(action)

        # 3. Predict from the same global vectors used during training.
        prediction = self.model.predict(projected, action_embedding)
        width = int(projected.shape[-1])
        if encoder.shape[-1] != width:
            raise ValueError(
                'Frozen stage probes require encoder and projector widths to '
                'match so samples remain one aligned feature tensor.'
            )
        return encoder, projected, prediction, width

    def _prepare_inputs(
        self,
        pixels: torch.Tensor,
        action_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the exact pixel and action preprocessing once."""
        pixels = self.prepare_pixels(pixels)
        action_history = self.normalize_action_history(
            _finite_or_zero(action_history).to(self.mean.device)
        )
        return pixels, action_history

    @torch.no_grad()
    def representation_features(
        self,
        pixels: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Return encoder/projector/predictor vectors as ``(B, 3, D)``."""
        # 1. Match training preprocessing before inspecting any model stage.
        pixels, action_history = self._prepare_inputs(pixels, action_history)

        # 2. Run the frozen pipeline once so all stage probes see one history.
        encoder, projector, predictor, visual_width = (
            self._encode_stage_history(pixels, action_history)
        )

        # 3. Read the last observed frame from current stages and the aligned
        # one-step-ahead vector from the predictor.
        stages = (encoder, projector, predictor)
        readouts = [
            _frame_feature(stage[:, -1:, :visual_width]) for stage in stages
        ]
        shapes = {tuple(readout.shape) for readout in readouts}
        if len(shapes) != 1:
            raise ValueError(
                f'Frozen probe stage shapes do not match: {sorted(shapes)}.'
            )
        return torch.stack(readouts, dim=1).float()

    @torch.no_grad()
    def location_features(
        self,
        pixels: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Return separate current and predicted vectors as ``(B, 2, D)``."""
        # 1. Reproduce the pixel and action preprocessing used in training.
        pixels, action_history = self._prepare_inputs(pixels, action_history)
        # 2. Run the frozen encoder and predictor over the aligned history.
        embedding, prediction, visual_width = self._encode_history(
            pixels, action_history
        )
        # 3. Read the last global vector from each history.
        current = _frame_feature(embedding[:, -1:, :visual_width])
        predicted = _frame_feature(prediction[:, -1:, :visual_width])
        # 4. Enforce a stable probe width before stacking both feature sources.
        if current.shape != predicted.shape:
            raise ValueError(
                'Location current/predicted shapes do not match: '
                f'{tuple(current.shape)} != {tuple(predicted.shape)}.'
            )
        width = int(current.shape[-1])
        if self._feature_width is None:
            self._feature_width = width
        elif self._feature_width != width:
            raise ValueError('Location feature width changed between batches.')
        return torch.stack((current, predicted), dim=1).float()

    @property
    def location_feature_width(self) -> int:
        """Return the visual width after the first feature extraction."""
        if self._feature_width is None:
            raise RuntimeError(
                'Run location_features once before reading its width.'
            )
        return self._feature_width


@contextmanager
def frozen_inference(*modules: nn.Module) -> Iterator[None]:
    """Run inference and restore every nested module's exact training flag."""
    # 1. Deduplicate nested modules because parents may share child objects.
    nested: list[nn.Module] = []
    seen: set[int] = set()
    for root in modules:
        for module in root.modules():
            if id(module) not in seen:
                seen.add(id(module))
                nested.append(module)
    # 2. Save each individual flag instead of assuming one root-wide mode.
    states = [module.training for module in nested]
    try:
        # 3. Assign flags directly so custom ``train`` implementations cannot
        # alter parameters or overwrite a child's independently tracked state.
        for module in nested:
            module.training = False
        with torch.inference_mode():
            yield
    finally:
        # 4. Restore the exact mixed train/eval state supplied by the caller.
        for module, training in zip(nested, states, strict=True):
            module.training = training


@contextmanager
def _preserve_global_torch_rng_state(
    device: torch.device,
) -> Iterator[None]:
    """Restore caller-owned CPU and CUDA generators after extraction."""
    cpu_state = torch.random.get_rng_state()
    cuda_device = device if device.type == 'cuda' else None
    cuda_state = (
        torch.cuda.get_rng_state(cuda_device)
        if cuda_device is not None
        else None
    )
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_device)


def _inference_autocast(
    device: torch.device, dtype: torch.dtype | None
) -> Any:
    if dtype is None:
        return nullcontext()
    supported = {
        'cpu': (torch.bfloat16,),
        'cuda': (torch.float16, torch.bfloat16),
    }
    if device.type not in supported or dtype not in supported[device.type]:
        choices = supported.get(device.type, ())
        raise ValueError(
            f'Unsupported autocast dtype {dtype} for {device.type}; '
            f'expected one of {choices or (None,)}.'
        )
    return torch.autocast(device_type=device.type, dtype=dtype)


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def extract_location_features(
    dataset: Dataset,
    extractor: nn.Module,
    *,
    device: torch.device | str = 'cpu',
    batch_size: int = 128,
    num_workers: int = 0,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract separate current/predicted vectors under explicit autocast."""
    # 1. Build a deterministic, ordered loader and output accumulators.
    device = torch.device(device)
    loader_generator = torch.Generator().manual_seed(0)
    feature_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    split_batches: list[torch.Tensor] = []
    sample_batches: list[torch.Tensor] = []
    # 2. Preserve caller RNG state while moving the frozen extractor to its device.
    with _preserve_global_torch_rng_state(device):
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            shuffle=False,
            persistent_workers=int(num_workers) > 0,
            pin_memory=device.type == 'cuda',
            generator=loader_generator,
        )
        extractor.to(device)
        with (
            frozen_inference(extractor),
            _inference_autocast(device, autocast_dtype),
        ):
            # 3. Extract and validate every batch before moving results to CPU.
            for batch in loader:
                batch = _move_batch(batch, device)
                features = extractor.location_features(
                    batch['pixels'],
                    batch['action_history'],
                )
                if (
                    features.ndim != 3
                    or features.shape[1] != len(FEATURE_SOURCES)
                    or features.shape[0] != batch['target'].shape[0]
                ):
                    raise ValueError(
                        'Location readout must return (batch, current/predicted, '
                        f'feature); got {tuple(features.shape)}.'
                    )
                if not torch.isfinite(features).all():
                    raise FloatingPointError(
                        'Location features contain NaN or infinity.'
                    )
                feature_batches.append(features.float().cpu())
                target_batches.append(batch['target'].float().cpu())
                split_batches.append(batch['split'].cpu())
                sample_batches.append(batch['sample_id'].cpu())
    # 4. Concatenate in sidecar order so sample IDs remain aligned with features.
    return tuple(
        torch.cat(values).numpy()
        for values in (
            feature_batches,
            target_batches,
            split_batches,
            sample_batches,
        )
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class FeatureCacheMetadata:
    """Compatibility contract for a frozen location-feature snapshot."""

    source_sha256: str
    checkpoint_sha256: str
    readout: str
    sample_sha256: str
    feature_width: int
    schema_version: int = FEATURE_CACHE_SCHEMA

    def validate(self) -> None:
        if self.schema_version != FEATURE_CACHE_SCHEMA:
            raise ValueError(
                f'Unsupported feature-cache schema {self.schema_version}.'
            )
        for name in (
            'source_sha256',
            'checkpoint_sha256',
            'sample_sha256',
        ):
            value = getattr(self, name)
            if not _is_sha256(value):
                raise ValueError(f'{name} must be a lowercase SHA-256 digest.')
        if self.readout != FEATURE_READOUT:
            raise ValueError(
                f'Expected readout {FEATURE_READOUT!r}, got {self.readout!r}.'
            )
        if self.feature_width <= 0:
            raise ValueError('feature_width must be positive.')


def snapshot_cache_path(
    output_dir: str | Path,
    checkpoint_sha256: str,
    *,
    batch_index: int | None = None,
    fit_end: bool = False,
) -> Path:
    """Return a deterministic, non-overwriting cache snapshot name."""
    if (batch_index is None) == (not fit_end):
        raise ValueError('Specify exactly one of batch_index or fit_end=True.')
    if batch_index is not None and batch_index < 0:
        raise ValueError('batch_index must be non-negative.')
    marker = 'fit_end' if fit_end else f'batch_{batch_index:09d}'
    path = (
        Path(output_dir)
        / f'location_features_v{FEATURE_CACHE_SCHEMA}_{marker}_'
        f'{checkpoint_sha256[:12]}.h5'
    )
    if path.exists() or path.with_suffix('.partial.h5').exists():
        raise FileExistsError(
            f'Location feature snapshot already exists: {path}'
        )
    return path


def write_feature_cache(
    path: str | Path,
    features: np.ndarray,
    targets: np.ndarray,
    split_codes: np.ndarray,
    sample_ids: np.ndarray,
    target_names: Sequence[str],
    metadata: FeatureCacheMetadata,
) -> Path:
    """Atomically publish one immutable HDF5 feature snapshot."""
    # 1. Normalize dtypes and validate every array against the cache contract.
    path = Path(path)
    metadata.validate()
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    splits = _split_codes(split_codes)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    count = len(features)
    if features.shape != (count, len(FEATURE_SOURCES), metadata.feature_width):
        raise ValueError(
            'Feature array must match samples x current/predicted x feature_width.'
        )
    if targets.shape != (count, len(target_names)):
        raise ValueError('Target array does not match samples x target_names.')
    if splits.shape != (count,) or sample_ids.shape != (count,):
        raise ValueError(
            'Split and sample ID arrays must match feature count.'
        )
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError('Feature cache contains NaN or infinity.')
    # 2. Reserve a fresh partial path so snapshots can never be overwritten.
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial.h5')
    if path.exists() or partial.exists():
        raise FileExistsError(
            f'Refusing to overwrite feature snapshot {path}.'
        )
    # 3. Write completely, then atomically publish; remove partials on failure.
    try:
        with h5py.File(partial, 'x') as h5:
            for name, value in asdict(metadata).items():
                h5.attrs[name] = value
            h5.attrs['target_names_json'] = json.dumps(list(target_names))
            h5.create_dataset('features', data=features, compression='gzip')
            h5.create_dataset('target', data=targets, compression='gzip')
            h5.create_dataset('split', data=splits)
            h5.create_dataset('sample_id', data=sample_ids)
            h5.flush()
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path


def load_feature_cache(
    path: str | Path,
    expected: FeatureCacheMetadata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load a feature cache only when all compatibility metadata matches."""
    # 1. Validate the requested contract before opening the stored snapshot.
    expected.validate()
    with h5py.File(path, 'r') as h5:
        # 2. Reconstruct and compare all provenance fields before reading arrays.
        actual = FeatureCacheMetadata(
            source_sha256=_decode(h5.attrs.get('source_sha256', '')),
            checkpoint_sha256=_decode(h5.attrs.get('checkpoint_sha256', '')),
            readout=_decode(h5.attrs.get('readout', '')),
            sample_sha256=_decode(h5.attrs.get('sample_sha256', '')),
            feature_width=int(h5.attrs.get('feature_width', -1)),
            schema_version=int(h5.attrs.get('schema_version', -1)),
        )
        if actual != expected:
            raise ValueError(
                'Location feature cache metadata mismatch: '
                f'expected {expected}, found {actual}.'
            )
        # 3. Return arrays only after the cache is proven compatible.
        names = tuple(json.loads(_decode(h5.attrs['target_names_json'])))
        return (
            h5['features'][:],
            h5['target'][:],
            h5['split'][:],
            h5['sample_id'][:],
            names,
        )


@dataclass(frozen=True)
class StandardizedRidge:
    """Closed-form multi-target ridge model with frozen standardization."""

    alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coefficient: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (
            np.asarray(features, dtype=np.float64) - self.x_mean
        ) / self.x_scale
        return (x @ self.coefficient) * self.y_scale + self.y_mean


def fit_standardized_ridge(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> StandardizedRidge:
    """Fit a standardized ridge regressor with a primal or dual solve."""
    # 1. Validate and standardize both features and multi-point targets.
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError(
            'Ridge inputs must be non-empty matrices with equal rows.'
        )
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('Ridge alpha must be finite and positive.')
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x_scale = np.maximum(x.std(axis=0), 1e-12)
    y_scale = np.maximum(y.std(axis=0), 1e-12)
    xs = (x - x_mean) / x_scale
    ys = (y - y_mean) / y_scale
    # 2. Solve the smaller primal or dual system for the same ridge objective.
    if xs.shape[1] <= xs.shape[0]:
        gram = xs.T @ xs + alpha * np.eye(xs.shape[1])
        coefficient = np.linalg.solve(gram, xs.T @ ys)
    else:
        gram = xs @ xs.T + alpha * np.eye(xs.shape[0])
        coefficient = xs.T @ np.linalg.solve(gram, ys)
    # 3. Store normalization with the coefficients for identical test readout.
    return StandardizedRidge(
        alpha=float(alpha),
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        coefficient=coefficient,
    )


def _point_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """Measure one named 2D point with coordinate MAE and mean x/y R2."""
    error = prediction - target
    residual = np.sum(error**2, axis=0)
    centered = target - target.mean(axis=0)
    total = np.sum(centered**2, axis=0)
    r2_by_dim = np.full(target.shape[1], np.nan, dtype=np.float64)
    variable = total > 0
    r2_by_dim[variable] = 1.0 - residual[variable] / total[variable]
    return {
        'mae': float(np.mean(np.abs(error))),
        'r2': float(np.nanmean(r2_by_dim)) if variable.any() else float('nan'),
    }


def regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    target_names: Sequence[str],
) -> dict[str, Any]:
    """Report only named-point metrics and their unweighted average."""
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if target.shape != prediction.shape or target.shape[1] != len(
        target_names
    ):
        raise ValueError(
            'Predictions, targets, and target names are incompatible.'
        )
    points: dict[str, dict[str, float]] = {}
    for index, name in enumerate(target_names):
        if not name.endswith('_x'):
            continue
        prefix = name[:-2]
        y_name = f'{prefix}_y'
        if y_name not in target_names:
            continue
        columns = [index, target_names.index(y_name)]
        points[prefix] = _point_metrics(
            target[:, columns], prediction[:, columns]
        )
    if not points:
        raise ValueError('Location targets contain no complete 2D points.')
    return {
        'points': points,
        'average': {
            metric: float(
                np.nanmean([values[metric] for values in points.values()])
            )
            for metric in ('mae', 'r2')
        },
    }


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, class_count: int
) -> dict[str, Any]:
    """Report accuracy, balanced accuracy, and per-class support."""
    target = np.asarray(targets, dtype=np.int64)
    prediction = np.asarray(predictions, dtype=np.int64)
    if (
        target.shape != prediction.shape
        or target.ndim != 1
        or len(target) == 0
    ):
        raise ValueError(
            'Classification targets must be equal non-empty vectors.'
        )
    if not np.isin(target, np.arange(class_count)).all():
        raise ValueError('Classification target is outside the class range.')
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (target, prediction), 1)
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.full(class_count, np.nan),
        where=support > 0,
    )
    return {
        'accuracy': float(np.mean(target == prediction)),
        'balanced_accuracy': float(np.nanmean(recall)),
        'chance_balanced_accuracy': 1.0 / class_count,
        'support': support.tolist(),
        'confusion_matrix': confusion.tolist(),
    }


def fit_ridge_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    class_count: int,
    alpha: float,
) -> dict[str, Any]:
    """Fit a standardized one-hot ridge readout and score its held-out argmax."""
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            'Classification requires non-empty train and test rows.'
        )
    labels = np.asarray(labels, dtype=np.int64)
    one_hot = np.eye(class_count, dtype=np.float64)[labels[train_mask]]
    model = fit_standardized_ridge(features[train_mask], one_hot, alpha)
    prediction = model.predict(features[test_mask]).argmax(axis=1)
    return classification_metrics(labels[test_mask], prediction, class_count)


_POSITIONAL_HUB_LABELS = (
    'single',
    'top',
    'upper_middle',
    'middle',
    'lower_middle',
    'bottom',
)


def _positional_hub_columns(
    target_names: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Resolve the non-semantic hub slots used by the dominant-motion probe."""
    pairs = []
    for label in _POSITIONAL_HUB_LABELS:
        x_name = f'tunnel_{label}_hub_x'
        y_name = f'tunnel_{label}_hub_y'
        if x_name in target_names and y_name in target_names:
            pairs.append(
                (target_names.index(x_name), target_names.index(y_name))
            )
    if not pairs:
        # Hidden or older sidecars may expose only one naming scheme. In that
        # case, order every hub by its stored columns and reject duplicate names.
        prefixes = [
            name[:-2]
            for name in target_names
            if name.startswith('tunnel_') and name.endswith('_hub_x')
        ]
        pairs = [
            (
                target_names.index(f'{prefix}_x'),
                target_names.index(f'{prefix}_y'),
            )
            for prefix in prefixes
        ]
    if not pairs:
        raise ValueError('Tunnel objective probe requires at least one hub.')
    return tuple(pairs)


def _nearest_hub_targets(
    targets: np.ndarray,
    target_names: Sequence[str],
    agent_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive nearest-hub identity, distance, and unit direction per sample."""
    agent_columns = _coordinate_columns(target_names, agent_prefix)
    hub_columns = _positional_hub_columns(target_names)
    agent = targets[:, agent_columns]
    hubs = np.stack([targets[:, columns] for columns in hub_columns], axis=1)
    displacement = hubs - agent[:, None]
    distances = np.linalg.norm(displacement, axis=-1)
    identity = distances.argmin(axis=1).astype(np.int64)
    rows = np.arange(len(targets))
    nearest_distance = distances[rows, identity]
    nearest_direction = displacement[rows, identity] / np.maximum(
        nearest_distance[:, None], 1e-8
    )
    return identity, nearest_distance[:, None], nearest_direction


def fit_tunnel_objective_probes(
    features: np.ndarray,
    targets: np.ndarray,
    split_codes: np.ndarray,
    target_names: Sequence[str],
    ridge_alpha: float,
) -> dict[str, dict[str, Any]]:
    """Probe the dominant objective: which tunnel is nearest and how to reach it."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    splits = _split_codes(split_codes)
    train = splits != SPLIT_TO_CODE['test']
    test = splits == SPLIT_TO_CODE['test']
    class_count = len(_positional_hub_columns(target_names))
    result: dict[str, dict[str, Any]] = {}

    # Align the predictor with the future agent state; the current readout uses
    # the final observed agent state from the same frozen history.
    prefixes = {
        'current_visual': 'agent_current',
        'predicted_visual': 'agent_future',
    }
    for index, source in enumerate(FEATURE_SOURCES):
        identity, distance, direction = _nearest_hub_targets(
            y, target_names, prefixes[source]
        )
        stage_features = x[:, index]
        distance_model = fit_standardized_ridge(
            stage_features[train], distance[train], ridge_alpha
        )
        direction_model = fit_standardized_ridge(
            stage_features[train], direction[train], ridge_alpha
        )
        distance_prediction = distance_model.predict(stage_features[test])
        direction_prediction = direction_model.predict(stage_features[test])
        denominator = np.maximum(
            np.linalg.norm(direction_prediction, axis=1), 1e-8
        )
        cosine = (
            np.sum(direction[test] * direction_prediction, axis=1)
            / denominator
        )
        result[source] = {
            'nearest_tunnel_identity': fit_ridge_classifier(
                stage_features,
                identity,
                train,
                test,
                class_count,
                ridge_alpha,
            ),
            'nearest_tunnel_distance': {
                'mae': float(
                    np.mean(np.abs(distance_prediction - distance[test]))
                ),
            },
            'nearest_tunnel_direction': {
                'component_mae': float(
                    np.mean(np.abs(direction_prediction - direction[test]))
                ),
                'mean_cosine': float(np.mean(np.clip(cosine, -1.0, 1.0))),
            },
        }
    return result


@dataclass(frozen=True)
class LocationProbeResult:
    """Held-out evaluation of one fixed probe per visual feature source."""

    ridge_alpha: float
    test_metrics: dict[str, dict[str, Any]]
    objective_metrics: dict[str, dict[str, Any]]
    sample_counts: dict[str, int]
    feature_width: int
    readout: str = FEATURE_READOUT


def fit_location_probe(
    features: np.ndarray,
    targets: np.ndarray,
    split_codes: np.ndarray,
    target_names: Sequence[str],
    ridge_alpha: float,
) -> tuple[dict[str, StandardizedRidge], LocationProbeResult]:
    """Fit one fixed linear map per feature source and test exactly once."""
    # 1. Validate the shared feature, target, and regularization contract.
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    alpha = float(ridge_alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            'Location-probe ridge_alpha must be finite and positive.'
        )
    if x.ndim != 3 or x.shape[1] != len(FEATURE_SOURCES):
        raise ValueError(
            'Location features must have shape '
            '(samples, current/predicted, width).'
        )
    if y.ndim != 2 or len(x) != len(y) or y.shape[1] != len(target_names):
        raise ValueError('Location features and targets are incompatible.')
    # 2. Combine fit and validation rows, preserving test as a one-shot holdout.
    splits = _split_codes(split_codes)
    masks = {name: splits == code for name, code in SPLIT_TO_CODE.items()}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError(
            'Fit, validation, and test splits must all be non-empty.'
        )
    train_mask = masks['fit'] | masks['validation']
    # 3. Give current and predicted features only their aligned agent target.
    current_columns = set(_coordinate_columns(target_names, 'agent_current'))
    future_columns = set(_coordinate_columns(target_names, 'agent_future'))
    geometry_columns = tuple(
        index
        for index in range(len(target_names))
        if index not in current_columns | future_columns
    )
    columns = {
        'current_visual': tuple(sorted(current_columns)) + geometry_columns,
        'predicted_visual': tuple(sorted(future_columns)) + geometry_columns,
    }
    # 4. Fit and evaluate one independent ridge map per feature source.
    models: dict[str, StandardizedRidge] = {}
    test_metrics: dict[str, dict[str, Any]] = {}
    for feature_index, source in enumerate(FEATURE_SOURCES):
        target_columns = columns[source]
        source_targets = y[:, target_columns]
        source_names = tuple(target_names[index] for index in target_columns)
        model = fit_standardized_ridge(
            x[train_mask, feature_index], source_targets[train_mask], alpha
        )
        models[source] = model
        test_metrics[source] = regression_metrics(
            source_targets[masks['test']],
            model.predict(x[masks['test'], feature_index]),
            source_names,
        )
    # 5. Return the fitted maps with their held-out metrics and sample counts.
    result = LocationProbeResult(
        ridge_alpha=alpha,
        test_metrics=test_metrics,
        objective_metrics=fit_tunnel_objective_probes(
            x, y, splits, target_names, alpha
        ),
        sample_counts={
            'train': int(train_mask.sum()),
            'test': int(masks['test'].sum()),
        },
        feature_width=int(x.shape[2]),
    )
    return models, result


def flatten_probe_metrics(
    result: LocationProbeResult,
) -> dict[str, float]:
    """Flatten held-out MAE/R2 into encoder and predictor histories."""
    namespaces = {
        'current_visual': 'location_probe_encoder',
        'predicted_visual': 'location_probe_predictor',
    }
    metrics: dict[str, float] = {}
    for source, source_metrics in result.test_metrics.items():
        source_prefix = f'{namespaces[source]}/test'
        for point, values in source_metrics['points'].items():
            for metric, score in values.items():
                metrics[f'{source_prefix}/{point}/{metric}'] = score
        for metric, score in source_metrics['average'].items():
            metrics[f'{source_prefix}/average/{metric}'] = score
        objective = result.objective_metrics[source]
        for metric in ('accuracy', 'balanced_accuracy'):
            metrics[f'{source_prefix}/nearest_tunnel_identity/{metric}'] = (
                objective['nearest_tunnel_identity'][metric]
            )
        metrics[f'{source_prefix}/nearest_tunnel_distance/mae'] = objective[
            'nearest_tunnel_distance'
        ]['mae']
        for metric, score in objective['nearest_tunnel_direction'].items():
            metrics[f'{source_prefix}/nearest_tunnel_direction/{metric}'] = (
                score
            )
    return metrics


def flatten_probe_summary(
    result: LocationProbeResult, prefix: str = 'location_probe'
) -> dict[str, float | int | str]:
    """Flatten static probe metadata for W&B run summary fields."""
    summary: dict[str, float | int | str] = {
        f'{prefix}/ridge_alpha': result.ridge_alpha,
        f'{prefix}/feature_width': result.feature_width,
        f'{prefix}/feature_readout': result.readout,
    }
    for split, count in result.sample_counts.items():
        summary[f'{prefix}/samples/{split}'] = count
    return summary
