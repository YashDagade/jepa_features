"""Finite-action geodesic reference with swept portals and rearm state.

Distances minimize pre-teleport Euclidean movement on a finite graph; teleport
jump length is excluded. They are not calibrated latent energies or exact
continuous-action optimal distances. The default one-pixel graph matches the
paper; use --spacing 0.5 for its refinement check.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from stable_worldmodel.envs.tunnel_rooms.env import room_bounds
from geometry import Reference, build_grid, common_references, make_env
from energy import save_overlays


def room_ids(points):
    return np.where(points[:, 0] < 112, 0, np.where(points[:, 1] < 112, 1, 2))


def build_graph(spacing, scenario, queries, refs):
    """Preserve swept triggers and the hidden rearm state on a finite action graph."""
    if spacing not in (1.0, 0.5):
        raise ValueError('Supported graph spacings are 1.0 and 0.5 pixels.')
    expected = {
        'num_leaves': 2,
        'speed': 5,
        'trigger_radius': 10,
        'landing_offset': 14,
        'rearm_distance': 28,
    }
    for key, value in expected.items():
        if scenario.get(key) != value:
            raise ValueError(f'This reference requires {key}={value}.')
    if scenario['pairing'] != 0:
        raise ValueError('The paper diagnostic uses pairing=0.')
    endpoints = np.asarray(scenario['endpoint_positions'], dtype=np.float32)
    bounds = np.asarray(
        [room_bounds(room, 2) for room in range(3)], dtype=np.float64
    )
    endpoint_rooms = np.array([0, 0, 1, 2])
    pairs = np.array([2, 3, 0, 1])
    landings = endpoints + np.array(
        [[-14, 0], [-14, 0], [14, 0], [14, 0]], dtype=np.float32
    )
    points, rooms, edge_start, edge_end = [], [], [], []
    offset = 0
    directions = [
        (x, y)
        for x, y in [
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
            (2, 1),
            (2, -1),
            (1, 2),
            (1, -2),
        ]
    ]
    for room, (xmin, xmax, ymin, ymax) in enumerate(bounds):
        xs = np.arange(xmin, xmax + 1e-6, spacing)
        ys = np.arange(ymin, ymax + 1e-6, spacing)
        xx, yy = np.meshgrid(xs, ys)
        p = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        ids = np.arange(offset, offset + len(p)).reshape(len(ys), len(xs))
        for dx, dy in directions:
            yy0, xx0 = np.indices(ids.shape)
            ok = (
                (xx0 + dx >= 0)
                & (xx0 + dx < len(xs))
                & (yy0 + dy >= 0)
                & (yy0 + dy < len(ys))
            )
            a = ids[yy0[ok], xx0[ok]]
            b = ids[yy0[ok] + dy, xx0[ok] + dx]
            edge_start.extend([a, b])
            edge_end.extend([b, a])
        points.append(p)
        rooms.append(np.full(len(p), room))
        offset += len(p)
    points = np.concatenate(points).astype(np.float32)
    rooms = np.concatenate(rooms)
    dense_count = len(points)
    extra = np.concatenate([queries, refs, landings, endpoints])
    extra_ids = []
    index = {tuple(p): i for i, p in enumerate(points)}
    new = []
    for p in extra:
        key = tuple(p)
        if key not in index:
            index[key] = dense_count + len(new)
            new.append(p)
        extra_ids.append(index[key])
    points = np.concatenate([points, np.asarray(new, dtype=np.float32)])
    rooms = room_ids(points)
    for i in range(dense_count, len(points)):
        candidates = np.flatnonzero((rooms[:dense_count] == rooms[i]))
        distances, nearest = cKDTree(points[candidates]).query(points[i], k=16)
        adjacent = candidates[nearest]
        assert np.abs(points[adjacent] - points[i]).max() <= 5
        edge_start.extend([np.full(len(adjacent), i), adjacent])
        edge_end.extend([adjacent, np.full(len(adjacent), i)])
    # Exact special positions close enough for a legal primitive action connect directly.
    special = np.unique(extra_ids)
    aa, bb = np.meshgrid(special, special, indexing='ij')
    ok = (
        (rooms[aa] == rooms[bb])
        & (np.max(np.abs(points[aa] - points[bb]), axis=-1) <= 5)
        & (aa != bb)
    )
    edge_start.append(aa[ok])
    edge_end.append(bb[ok])
    # A zero action inside an armed portal must still trigger teleportation.
    edge_start.append(np.arange(len(points)))
    edge_end.append(np.arange(len(points)))
    a = np.concatenate(edge_start)
    b = np.concatenate(edge_end)
    delta = points[b].astype(float) - points[a]
    lengths = np.linalg.norm(delta, axis=1)
    assert np.abs(delta).max() <= 5
    dist = np.empty((len(a), 4))
    denom = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
    for k, p in enumerate(endpoints):
        fraction = np.clip(
            np.sum((p - points[a]) * delta, axis=1) / denom, 0, 1
        )
        closest = points[a] + fraction[:, None] * delta
        dist[:, k] = np.linalg.norm(closest - p, axis=1)
        dist[rooms[a] != endpoint_rooms[k], k] = np.inf
    # State zero is unlocked; state k+1 locks endpoint k until its 28px rearm boundary.
    node = np.full((5, len(points)), -1, dtype=np.int32)
    node[0] = np.arange(len(points))
    node_points = [np.arange(len(points))]
    locks = [np.full(len(points), -1)]
    next_id = len(points)
    for k in range(4):
        subset = np.flatnonzero(
            (rooms == endpoint_rooms[k])
            & (np.linalg.norm(points - endpoints[k], axis=1) < 28)
        )
        node[k + 1, subset] = np.arange(next_id, next_id + len(subset))
        next_id += len(subset)
        node_points.append(subset)
        locks.append(np.full(len(subset), k))
    source, destination, cost, movement = [], [], [], []
    landing_ids = np.asarray(
        extra_ids[len(queries) + len(refs) : len(queries) + len(refs) + 4]
    )
    for layer in range(5):
        selected = np.flatnonzero(node[layer, a] >= 0)
        aa = a[selected]
        bb = b[selected]
        distances = dist[selected].copy()
        if layer:
            distances[:, layer - 1] = np.inf
        hit = np.argmin(distances, axis=1)
        trigger = distances[np.arange(len(hit)), hit] <= 10
        target_position = bb.copy()
        target_layer = np.full(len(aa), layer)
        if layer:
            rearm = (
                np.linalg.norm(points[bb] - endpoints[layer - 1], axis=1) >= 28
            )
            target_layer[rearm] = 0
        target_position[trigger] = landing_ids[pairs[hit[trigger]]]
        target_layer[trigger] = pairs[hit[trigger]] + 1
        source.append(node[layer, aa])
        destination.append(node[target_layer, target_position])
        cost.append(lengths[selected])
        movement.append(bb)
    source = np.concatenate(source)
    destination = np.concatenate(destination)
    cost = np.concatenate(cost)
    movement = np.concatenate(movement)
    assert (destination >= 0).all()
    # Keep the cheapest actual primitive move for each directed state edge.
    order = np.lexsort((cost, destination, source))
    source = source[order]
    destination = destination[order]
    cost = cost[order]
    movement = movement[order]
    keep = np.r_[
        True,
        (source[1:] != source[:-1]) | (destination[1:] != destination[:-1]),
    ]
    source = source[keep]
    destination = destination[keep]
    cost = cost[keep]
    movement = movement[keep]
    graph = csr_matrix((cost, (source, destination)), shape=(next_id, next_id))
    result = []
    predecessors = []
    ref_ids = np.asarray(extra_ids[len(queries) : len(queries) + len(refs)])
    for goal in ref_ids[2:4]:
        targets = node[:, goal]
        targets = targets[targets >= 0]
        distances, predecessor, _ = dijkstra(
            graph.T.tocsr(),
            directed=True,
            indices=targets,
            min_only=True,
            return_predecessors=True,
        )
        assert np.isfinite(distances[node[0, extra_ids[: len(queries)]]]).all()
        assert distances[node[0, goal]] == 0
        result.append(distances)
        predecessors.append(predecessor)
    return dict(
        points=points,
        node_point=np.concatenate(node_points),
        node_lock=np.concatenate(locks),
        source=source,
        destination=destination,
        cost=cost,
        movement=movement,
        distances=np.stack(result),
        predecessors=np.stack(predecessors),
        node=node,
        query_ids=np.asarray(extra_ids[: len(queries)]),
        reference_ids=ref_ids,
        landing_ids=landing_ids,
        endpoint_ids=np.asarray(extra_ids[-4:]),
        spacing=np.array(spacing),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument(
        '--samples', type=Path, required=True, help='energy.npz from energy.py'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('outputs/geodesic')
    )
    parser.add_argument('--spacing', type=float, default=1.0)
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text())
    with np.load(args.samples) as samples:
        queries = samples['grid_positions'].copy()
        refs = samples['reference_positions'].copy()
        indices = samples['grid_indices'].copy()
        grid_size = samples['squared_projected_distance'].shape[-1]
    env = make_env(scenario)
    try:
        expected_grid = build_grid(env, grid_size)
        expected_positions = np.stack([p for _, _, p in expected_grid])
        expected_indices = np.asarray([(y, x) for y, x, _ in expected_grid])
        expected_refs = np.stack(
            [ref.position for ref in common_references(scenario)]
        )
        if not (
            np.array_equal(queries, expected_positions)
            and np.array_equal(indices, expected_indices)
            and np.array_equal(refs, expected_refs)
        ):
            raise ValueError(
                'Samples do not match this scenario and reference protocol.'
            )
    finally:
        env.close()
    graph = build_graph(args.spacing, scenario, queries, refs)
    costs = graph['distances'][:, graph['query_ids']]
    field = np.full((2, grid_size, grid_size), np.nan)
    field[:, indices[:, 0], indices[:, 1]] = costs
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / 'geodesic.npz',
        geodesic_distance=field,
        grid_positions=queries,
        reference_positions=refs,
        grid_indices=indices,
        reference_distances=graph['distances'][:, graph['reference_ids']],
    )
    env = make_env(scenario)
    try:
        references = [
            Reference(endpoint=i, room=i - 1, position=refs[i]) for i in (2, 3)
        ]
        scale = save_overlays(env, references, field, args.output)
    finally:
        env.close()
    (args.output / 'method.json').write_text(
        json.dumps(
            {
                'units': 'pixels of pre-teleport movement',
                'spacing': args.spacing,
                'state': 'position and locked endpoint; queries initially unlocked',
                'goal': 'exact reference point with any lock state; not the success radius',
                'scale': scale,
                'directions': 16,
                'approximation': 'finite action graph with swept triggers and exact portal landing',
            },
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
