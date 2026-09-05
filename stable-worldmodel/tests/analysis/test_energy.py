"""Static-energy and finite-graph regression checks for the paper scenario."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parents[2] / 'scripts' / 'analysis'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ANALYSIS))
    return tuple(
        importlib.import_module(name)
        for name in ('geometry', 'energy', 'geodesic')
    )


def test_variants_share_valid_centers_and_reference_goals(modules):
    geometry, _, _ = modules
    grids, references = [], []
    for variant in ('color', 'shape', 'palette'):
        scenario = json.loads(
            (ANALYSIS / f'paper_{variant}_scenario.json').read_text()
        )
        env = geometry.make_env(scenario)
        try:
            grid = geometry.build_grid(env, 20)
            assert len(grid) == 210
            refs = geometry.common_references(scenario)
            for ref in refs:
                assert env.room_index(ref.position) == ref.room
                assert geometry.is_good_reference(
                    env,
                    ref.position,
                    env.endpoint_positions[ref.endpoint],
                    ref.room,
                )
            grids.append(np.stack([p for _, _, p in grid]))
            references.append(np.stack([r.position for r in refs]))
        finally:
            env.close()
    for i in (1, 2):
        np.testing.assert_array_equal(grids[0], grids[i])
        np.testing.assert_array_equal(references[0], references[i])


def test_energy_is_summed_squared_projected_distance(modules):
    _, energy, _ = modules
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    squared, cosine = energy.distances(features, 2)
    np.testing.assert_array_equal(squared, [[0.0, 5.0], [2.0, 5.0]])
    np.testing.assert_allclose(cosine, [[1.0, 2**-0.5], [0.0, 2**-0.5]])


def test_geodesic_matches_archived_query_distances(modules):
    geometry, _, geodesic = modules
    scenario = json.loads((ANALYSIS / 'paper_color_scenario.json').read_text())
    env = geometry.make_env(scenario)
    try:
        queries = np.stack([p for _, _, p in geometry.build_grid(env, 20)])
        refs = np.stack(
            [r.position for r in geometry.common_references(scenario)]
        )
    finally:
        env.close()
    graph = geodesic.build_graph(1.0, scenario, queries, refs)
    expected = json.loads(
        Path(__file__).with_name('geodesic_reference.json').read_text()
    )
    np.testing.assert_allclose(
        graph['distances'][:, graph['query_ids']], expected, rtol=0, atol=1e-10
    )
    reference_cost = graph['distances'][:, graph['reference_ids']]
    assert reference_cost[0, 0] < reference_cost[0, 1]
    assert reference_cost[1, 1] < reference_cost[1, 0]
    with pytest.raises(ValueError, match='speed'):
        geodesic.build_graph(1.0, dict(scenario, speed=2), queries, refs)
