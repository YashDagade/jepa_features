from pathlib import Path

import h5py
import numpy as np

from scripts.data.audit_objective_exposure import audit_dataset


def _write_dataset(path: Path, *, chaser: bool) -> None:
    positions = np.asarray(
        [[0, 0], [1, 0], [2, 0], [3, 0], [100, 0], [99, 0], [98, 0], [97, 0]],
        dtype=np.float32,
    )
    with h5py.File(path, 'w') as source:
        source.create_dataset('ep_offset', data=np.asarray([0], np.int64))
        source.create_dataset('ep_len', data=np.asarray([8], np.int64))
        source.create_dataset('pos_agent', data=positions)
        if chaser:
            source.create_dataset(
                'pos_blue', data=np.tile([10, 0], (len(positions), 1))
            )
            events = np.zeros(len(positions), dtype=bool)
            events[4] = True
            source.create_dataset('event', data=events)


def test_chaser_audit_separates_chase_teleport_and_other(tmp_path):
    path = tmp_path / 'chaser.h5'
    _write_dataset(path, chaser=True)

    result = audit_dataset(
        path, environment='chaser', frameskip=1, num_steps=3
    )

    # Every non-event move reduces red-blue distance, including after teleport.
    assert result['raw_counts'] == {'chase': 6, 'teleport': 1, 'other': 0}
    assert result['chase_to_teleport_ratio'] == 6.0
    assert result['loader']['legal_clips'] == 6
    assert result['loader']['teleport_predicted_interval_exposure'] == 2 / 12
    assert result['loader']['teleport_clip_exposure'] == 2 / 6


def test_tunnel_audit_uses_next_entered_portal_as_approach_target(tmp_path):
    path = tmp_path / 'tunnel.h5'
    _write_dataset(path, chaser=False)

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
