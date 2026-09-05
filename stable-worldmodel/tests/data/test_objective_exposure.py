from pathlib import Path

import h5py
import numpy as np

from scripts.data.audit_objective_exposure import audit_dataset


def _write_dataset(path: Path) -> None:
    positions = np.asarray(
        [[0, 0], [1, 0], [2, 0], [3, 0], [100, 0], [99, 0], [98, 0], [97, 0]],
        dtype=np.float32,
    )
    with h5py.File(path, 'w') as source:
        source.create_dataset('ep_offset', data=np.asarray([0], np.int64))
        source.create_dataset('ep_len', data=np.asarray([8], np.int64))
        source.create_dataset('pos_agent', data=positions)



def test_tunnel_audit_uses_next_entered_portal_as_approach_target(tmp_path):
    path = tmp_path / 'tunnel.h5'
    _write_dataset(path)

    result = audit_dataset(
        path,
        environment='tunnel',
        frameskip=1,
        num_steps=3,
        teleport_threshold=30,
    )

    assert result['raw_counts'] == {'approach': 3, 'teleport': 1, 'other': 3}
    assert result['approach_to_teleport_ratio'] == 3.0
    assert result['loader']['approach_clip_exposure'] == 3 / 6
