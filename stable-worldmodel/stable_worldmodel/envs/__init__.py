"""Gymnasium registration for the Tunnel Rooms benchmark."""

from gymnasium.envs import registration

WORLDS = set()
DISCRETE_WORLDS = set()


def register(id, entry_point, **kwargs):
    """Register one benchmark and expose its ID to collection utilities."""
    registration.register(id=id, entry_point=entry_point, **kwargs)
    WORLDS.add(id)


register(
    id='swm/TunnelRooms-v1',
    entry_point='stable_worldmodel.envs.tunnel_rooms.env:TunnelRoomsEnv',
)



__all__ = ['DISCRETE_WORLDS', 'WORLDS', 'register']
