import json
import websockets
from typing import TYPE_CHECKING
# if TYPE_CHECKING:
from slither_env import GameConnection
from websockets.asyncio.server import ServerConnection

async def handle_client(websocket: ServerConnection, connection: GameConnection):
    print(f"Client connected from {websocket.remote_address}")
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "update":
                payload = data["payload"]
                connection.put_state(payload)

                action = connection.latest_action
                if action is None:
                    action = [0.0, 0.0, 0.0]

                accelerate = 1 if action[2] > 0.9 else 0

                action_message = {
                    "type": "update",
                    "payload": {
                        "xt": float(action[0]),
                        "yt": float(action[1]),
                        "acceleration": accelerate
                    }
                }
                await websocket.send(json.dumps(action_message))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
