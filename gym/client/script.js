// ==UserScript==
// @name         Slither.io Gym
// @include      http://slither.com/io
// @author       James Stratford
// @description  Reinforcement learning for the Slither.io game
// @match        http://slither.com/io
// @run-at       document-end
// @grant        GM_info
// @grant        unsafeWindow
// @version      0.3.0
// ==/UserScript==

let slither_xt = 0;
let slither_yt = 0;
let slither_acceleration = 0;

// --- WebSocket Initialization ---
(function () {
    const socket = new WebSocket('ws://127.0.0.1:10043');

    // WebSocket Handlers
    socket.addEventListener('open', () => {
        console.log('Connected to WebSocket server');
        socket.send(JSON.stringify({ type: 'init', message: 'Hello Server!' }));
    });

    socket.addEventListener('message', (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'update') {
            slither_xt = data.payload.xt;
            slither_yt = data.payload.yt;
            slither_acceleration = data.payload.acceleration;
        }
    });

    socket.addEventListener('close', () => console.log('Disconnected from WebSocket server'));
    socket.addEventListener('error', (error) => console.error('WebSocket error:', error));

    window.sendMessage = (type, payload) => {
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type, payload }));
        } else {
            console.error('WebSocket is not open. Ready state: ', socket.readyState);
        }
    };
})();

// --- Game State Variables ---
let frame = 0;
let sent_death = false;
let last_state = {};
let food_eaten = 0;
let last_size = 0;
let last_xx = 0;
let last_yy = 0;
let inferred_death = false;
let marked_dead_slithers = [];

// --- Utility Functions ---
const calculateDistance = (x1, y1, x2, y2) => Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));

// Process the top N closest body parts across all snakes
const processTopBodyParts = (slithers, referenceX, referenceY, topN = 100) => {
    let allParts = [];

    slithers.forEach((o_slither) => {
        if (o_slither.id === slither.id) return;
        o_slither.gptz.forEach((part) => {
            const dist = calculateDistance(part.xx, part.yy, referenceX, referenceY);
            allParts.push({ x: part.xx, y: part.yy, dist, size: o_slither.size });
        });
    });

    // Sort parts by distance and select the top N
    allParts.sort((a, b) => a.dist - b.dist);
    return allParts.slice(0, topN);
};

// Process food locations
const processFood = (foods, referenceX, referenceY, maxDistance = 1000) => {
    food_locations = [];
    for (let i = 0; i < foods_c; i++) {
        const dist = Math.sqrt(Math.pow(foods[i].xx - referenceX, 2) + Math.pow(foods[i].yy - referenceY, 2));
        if (dist < maxDistance) {
            food_locations.push({
                x: foods[i].xx,
                y: foods[i].yy,
                value: foods[i].gr,
                dist: dist
            });
        }
    }
    return food_locations;
};

// Process preys
const processPreys = (preys, referenceX, referenceY, maxDistance = 1000) => {
    preys_locations = [];
    for (let i = 0; i < preys.length; i++) {
        const dist = calculateDistance(preys[i].xx, preys[i].yy, referenceX, referenceY);
        if (dist < maxDistance) {
            preys_locations.push({ x: preys[i].xx, y: preys[i].yy, dist });
        }
    }
    return preys_locations;
};

// Process other slithers
const processOtherSlithers = (slithers, currentSnakeId, referenceX, referenceY, maxDistance = 1000) => {
    // Decrement TTL for each marked dead slither
    marked_dead_slithers.forEach((item) => item.ttl -= 1);
    marked_dead_slithers = marked_dead_slithers.filter((item) => item.ttl > 0);

    return slithers
        .filter((slither) => slither.id !== currentSnakeId)
        .map((other_slither) => {
            if (marked_dead_slithers.some((item) => item.id === other_slither.id && item.ttl > 0)) {
                return null;
            }
            if (other_slither.dead) {
                marked_dead_slithers.push({ id: other_slither.id, ttl: 60 });
            }
            const dist = calculateDistance(other_slither.xx, other_slither.yy, referenceX, referenceY);
            if (dist < maxDistance) {
                parts = other_slither.gptz.map((part) => ({
                    x: part.xx,
                    y: part.yy,
                    dist: calculateDistance(part.xx, part.yy, referenceX, referenceY)
                }));
                var sct = other_slither.sct + other_slither.rsc;
                let score = Math.floor((fpsls[sct] + other_slither.fam / fmlts[sct] - 1) * 15 - 5) / 1;
                return {
                    x: other_slither.xx,
                    y: other_slither.yy,
                    ang: other_slither.ang,
                    parts,
                    size: score,
                    dist,
                    dead: other_slither.dead,
                };
            }
            return null;
        })
        .filter((slither) => slither !== null);
};

const processClosestEnemyByPartsToHead = (slithers, referenceX, referenceY, maxDistance = 2000) => {
    return slithers.filter((slither) => slither.id !== slither.id)
        .map((other_slither) => {
            const parts = other_slither.parts.map((part) => {
                if (part.dist > maxDistance) return null;
                return {
                    x: part.xx,
                    y: part.yy,
                    dist: calculateDistance(part.xx, part.yy, referenceX, referenceY)
                };
            });
            parts.sort((a, b) => a.dist - b.dist);
            return other_slither;
        })
};

let connected = true;
const dec_connect = () => {
    connect();
    connected = true;
};

// --- Main Redraw Hook ---
const hk_redraw = () => {
    if (!slither || !connected) return;

    // Handle snake death
    if (sent_death && slither.xx !== last_xx && slither.yy !== last_yy) {
        sent_death = false;
    }
    if (!inferred_death && !sent_death && slither.xx === last_xx && slither.yy === last_yy) {
        inferred_death = true;
        last_size = 0;
    }

    last_xx = slither.xx;
    last_yy = slither.yy;

    if (inferred_death && !sent_death) {
        window.sendMessage('update', { dead: true, ...last_state });
        sent_death = true;
        inferred_death = false;
        last_size = 11;
        connected = false;
        setTimeout(dec_connect, 5000);
        return;
    }

    // Update game state
    if (++frame % 10 === 0) {
        frame = 0;

        parts = slither.gptz.map((part) => ({
            x: part.xx,
            y: part.yy,
            dist: calculateDistance(part.xx, part.yy, slither.xx, slither.yy),
            size: slither.size,
        }));
        var sct = slither.sct + slither.rsc;
        let score = Math.floor((fpsls[sct] + slither.fam / fmlts[sct] - 1) * 15 - 5) / 1;
        const slither_details = {
            dead: false,
            x: slither.xx,
            y: slither.yy,
            parts,
            ang: slither.ang,
            size: score,
            food_eaten,
        };

        food_eaten = score - last_size;
        last_size = score;

        const other_slithers = processOtherSlithers(slithers, slither.id, slither.xx, slither.yy, 2000);
        const food_locations = processFood(foods, slither.xx, slither.yy);
        const preys_locations = processPreys(preys, slither.xx, slither.yy);

        // Get top 100 closest body parts
        const top_body_parts = processTopBodyParts(slithers, slither.xx, slither.yy, 200);

        const target_slither = processClosestEnemyByPartsToHead(slithers, slither.xx, slither.yy, 200);
        const target_slither_details = {
            x: target_slither.xx,
            y: target_slither.yy,
            parts: target_slither.gptz,
            ang: target_slither.ang,
            size: target_slither.size,
        };

        const gameState = {
            slither: slither_details,
            target_slither: target_slither_details,
            foods: food_locations,
            preys: preys_locations,
            others: other_slithers,
            top_body_parts, // Add top 100 closest body parts
        };

        window.sendMessage('update', gameState);
        last_state = gameState;
    }

    const coef = 10000;
    xm = slither_xt * coef;
    ym = slither_yt * coef;
    setAcceleration(slither_acceleration ? 1 : 0);
};

// --- Inject Bot Logic ---
const injectBot = () => {
    if (typeof redraw !== "undefined") {
        const oldRedraw = redraw;
        redraw = () => {
            hk_redraw();
            oldRedraw();
        };
        console.log("Injected bot logic");
        window.onmousemove = null;
    } else {
        setTimeout(injectBot, 1000);
        console.log("Retrying bot injection...");
    }
};

// Inject and start
injectBot();
