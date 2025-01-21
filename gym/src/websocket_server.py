from typing import Awaitable, Callable
import websockets
from slither_env import GameConnection
from utils import handle_client as default_handle_client


async def start_server(
    connection: GameConnection,
    *,
    handle_client: Callable[
        [websockets.ServerProtocol, GameConnection], Awaitable[None]
    ] = default_handle_client,
    **kwargs
):
    print("Starting WebSocket server...")
    try:
        server = await websockets.serve(
            lambda ws: handle_client(ws, connection, **kwargs),
            "127.0.0.1",
            10043,
            ping_interval=20,
            ping_timeout=10,
        )
        print("WebSocket server started at ws://127.0.0.1:10043")
        await server.wait_closed()
    except Exception as e:
        print(f"Failed to start WebSocket server: {e}")
