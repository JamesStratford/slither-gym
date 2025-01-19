// ==UserScript==
// @name Slither.io Gym
// @include     http://slither.com/io
// @author       James Stratford
// @description  Reinforcement learning for the Slither.io game
// @match       http://slither.com/io
// @run-at document-end
// @grant    GM_info
// @grant         unsafeWindow
// @version 0.1.0
// ==/UserScript==
slither_xt = 0;
slither_yt = 0;
slither_acceleration = 0;

(function () {
    // Create a new WebSocket connection
    const socket = new WebSocket('ws://127.0.0.1:8765');

    // Connection opened
    socket.addEventListener('open', function (event) {
        console.log('Connected to WebSocket server');
        // You can send a message to the server if needed
        socket.send(JSON.stringify({ type: 'init', message: 'Hello Server!' }));
    });

    // Listen for messages
    socket.addEventListener('message', function (event) {
        const data = JSON.parse(event.data);
        if (data.type === 'update') {
            slither_xt = data.payload.xt;
            slither_yt = data.payload.yt;
            slither_acceleration = data.payload.acceleration;
        }
    });

    // Connection closed
    socket.addEventListener('close', function (event) {
        console.log('Disconnected from WebSocket server');
    });

    // Handle errors
    socket.addEventListener('error', function (error) {
        console.error('WebSocket error:', error);
    });

    // Example function to send a message to the server
    function sendMessage(type, payload) {
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type, payload }));
        } else {
            console.error('WebSocket is not open. Ready state: ', socket.readyState);
            return false;
        }
    }

    // Expose the sendMessage function to the global scope for testing
    window.sendMessage = sendMessage;

    return true;
})();

frame = 0;
sent_death = false;
last_state = {}
food_eaten = 0;
last_size = 0;
last_xx = 0;
last_yy = 0;
inferred_death = false;
hk_redraw = () => {
    if (sent_death && slither && slither.xx !== last_xx && slither.yy !== last_yy) {
        sent_death = false;
    }
    if (slither && !inferred_death && !sent_death) {
        if (last_xx == slither.xx && last_yy == slither.yy) {
            inferred_death = true;
        }

        last_xx = slither.xx;
        last_yy = slither.yy;
        sent_death = false;
        var coef = 10000;

        if (slither.fam > last_size) {
            food_eaten += slither.fam - last_size;
        }
        last_size = slither.fam;

        if (++frame % 10 === 0) {
            frame = 0;
            my_parts = []
            slither.pts.forEach(part => {
                my_parts.push({
                    x: part.xx - slither.xx,
                    y: part.yy - slither.yy,
                    dist: Math.sqrt(Math.pow(part.xx - slither.xx, 2) + Math.pow(part.yy - slither.yy, 2)),
                })
            })

            slither_details = {
                dead: false,
                x: slither.xx,
                y: slither.yy,
                xm: xm,
                ym: ym,
                parts: my_parts,
                size: slither.fam,
                food_eaten: food_eaten
            }
            food_eaten = 0;

            other_slithers = [];
            slithers.forEach(other_slither => {
                if (other_slither.id !== slither.id) {
                    parts = []
                    other_slither.pts.forEach(part => {
                        dist = Math.sqrt(Math.pow(part.xx - slither.xx, 2) + Math.pow(part.yy - slither.yy, 2))
                        if (dist < 1000) {
                            parts.push({
                                x: part.xx - slither.xx,
                                y: part.yy - slither.yy,
                                dist: dist
                            });
                        }
                    });

                    const slitherDist = Math.sqrt(Math.pow(other_slither.xx - slither.xx, 2) + Math.pow(other_slither.yy - slither.yy, 2));
                    if (slitherDist < 1000) {
                        other_slithers.push({
                            x: other_slither.xx,
                            y: other_slither.yy,
                            parts: parts,
                            dist: slitherDist
                        });
                    }
                }
            });

            food_locations = [];
            for (let i = 0; i < foods_c; i++) {
                const dist = Math.sqrt(Math.pow(foods[i].xx - slither.xx, 2) + Math.pow(foods[i].yy - slither.yy, 2));
                if (dist < 1000) {
                    food_locations.push({
                        x: foods[i].xx - slither.xx,
                        y: foods[i].yy - slither.yy,
                        value: foods[i].gr,
                        dist: dist
                    });
                }
            }

            preys_locations = [];
            preys.forEach(prey => {
                const dist = Math.sqrt(Math.pow(prey.xx - slither.xx, 2) + Math.pow(prey.yy - slither.yy, 2));
                if (dist < 1000) {
                    preys_locations.push({
                        x: prey.xx - slither.xx,
                        y: prey.yy - slither.yy,
                        dist: dist
                    });
                }
            });

            window.sendMessage('update', {
                slither: slither_details,
                foods: food_locations,
                preys: preys_locations,
                others: other_slithers
            });
            last_state = {
                slither: slither_details,
                foods: food_locations,
                preys: preys_locations,
                others: other_slithers
            }
        }
        xm = slither_xt * coef;
        ym = slither_yt * coef;
        setAcceleration(slither_acceleration ? 1 : 0);
    }
    else if (!slither) {
        return;
    }
    else if ((inferred_death && !sent_death)) {
        window.sendMessage('update', {
            dead: true,
            ...last_state
        });
        sent_death = true;
        inferred_death = false
        setTimeout(connect, 5000);
    }
}

const injectBot = () => {
    if (typeof (redraw) !== "undefined") {
        const oldredraw = redraw;
        redraw = () => {
            hk_redraw();
            oldredraw();
        };
        console.log("injected");
        window.onmousemove = null;
    } else {
        setTimeout(injectBot, 1000);
        console.log("retrying");
    }
};

injectBot();
setTimeout(() => {
    connect();
}, 5000);