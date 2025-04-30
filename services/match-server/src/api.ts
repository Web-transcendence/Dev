import Fastify from 'fastify';
import fastifyWebsocket from '@fastify/websocket';
import { WebSocket } from "ws";
import { z } from "zod";

const inputSchema = z.object({ state: z.string(), key: z.string() });
const initSchema = z.object({ nick: z.string(), room: z.number() });
const readySchema = z.object({ mode: z.string() });

class Ball {
    x: number;
    y: number;
    angle: number;
    speed: number;
    ispeed: number;
    ospeed: number;
    radius: number;
    color: string;
    constructor(x: number, y: number, angle: number, speed: number, radius: number, color: string) {
        this.x = x;
        this.y = y;
        this.angle = angle;
        this.speed = speed;
        this.ispeed = speed;
        this.ospeed = speed;
        this.radius = radius;
        this.color = color;
    }
    toJSON() {
        return {type: "Ball", x: this.x, y: this.y,radius: this.radius, color: this.color};
    }
}

class Paddle {
    x: number;
    y: number;
    width: number;
    height: number;
    speed: number;
    color: string;
    score: string;
    constructor(x: number, y: number, width: number, height: number, speed: number, color: string) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.speed = speed;
        this.color = color;
        this.score = "0";
    }
    toJSON() {
        return {type: "Paddle", x: this.x, y: this.y, width: this.width, height: this.height, color: this.color, score: this.score};
    }
}

class keyInput {
    arrowUp: boolean = false;
    arrowDown: boolean = false;
    w: boolean = false;
    s: boolean = false;
}

class Hazard {
    x: number;
    y: number = 0;
    speed: number;
    type: string;
    constructor (x: number, speed: number, type: string) {
        this.x = x;
        this.speed = speed;
        this.type = type;
    }
    toJSON () {
        return {x: this.x, y: this.y, type: this.type};
    }
}

class gameState {
    state: number = 0;
    start: boolean = false;
    maxScore: string = "6";
    score1: string = "0";
    score2: string = "0";
    hazard: Hazard = new Hazard(0, 0, "Default");
    timer: Timer = new Timer (0, 3);
    toJSON() {
        return {type: "Game", state: this.state, start: this.start, score1: this.score1, score2: this.score2, hazard: this.hazard, timer: this.timer};
    }
}

class Room {
    id: number;
    players: Player[] = [];
    constructor (id: number, player: Player) {
        this.id = id;
        this.players.push(player);
    }
}

class Player {
    name: string = "Default";
    id: number;
    ws: WebSocket;
    paddle: Paddle = new Paddle(0, 400, 20, 200, 10, "#fcc800");
    input: keyInput = new keyInput();
    constructor(id: number, ws: WebSocket) {
        this.id = id;
        this.ws = ws;
    }
}

class Timer {
    timeLeft: number;
    started: boolean = false;
    intervalId: ReturnType<typeof setInterval> | null = null;
    constructor(minutes: number, seconds: number) {
        this.timeLeft = minutes * 60 + seconds;
    }
    start() {
        this.started = true;
        this.intervalId = setInterval(() => {
            if (this.timeLeft > 0)
                this.timeLeft--;
            else
                this.stop();
        }, 1000);
    }
    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
            this.started = false;
        }
    }
    toJSON() {
        return {timeLeft: this.timeLeft, started: this.started};
    }
}

let rooms: Room[] = [];
const ids = new Set<number>();

function generateId() {
    let newId: number;
    do {
        newId = Math.floor(Math.random() * 10000);
    } while (ids.has(newId));
    ids.add(newId);
    return (newId);
}

function inputHandler(key: string, state: string, input: keyInput) {
    let down = state === "down";
    if (key === "w")
        input.w = down;
    if (key === "s")
        input.s = down;
    if (key === "ArrowUp")
        input.arrowUp = down;
    if (key === "ArrowDown")
        input.arrowDown = down;
}

function resetInput(Input: keyInput) {
    Input.arrowUp = false;
    Input.arrowDown = false;
    Input.w = false;
    Input.arrowUp = false;
}

function resetGame(ball: Ball, lPaddle: Paddle, rPaddle: Paddle, game: gameState, lInput: keyInput, rInput: keyInput) {
    game.start = false;
    game.timer = new Timer (0, 2);
    if (ball.x < 0)
        ball.angle = Math.PI;
    else
        ball.angle = 0;
    ball.x = 0.5 * 1200;
    ball.y = 0.5 * 800;
    resetHazard(lPaddle, rPaddle, ball)
    ball.speed = ball.ispeed;
    lPaddle.y = 0.5 * 800;
    rPaddle.y = 0.5 * 800;
    game.hazard.type = "Default";
    if (rPaddle.score === game.maxScore || lPaddle.score === game.maxScore) {
        game.state = 2;
        game.score1 = lPaddle.score;
        game.score2 = rPaddle.score;
        rPaddle.score = "0";
        lPaddle.score = "0";
    }
    resetInput(lInput);
    resetInput(rInput);
}

function norAngle(ball: Ball) {
    if (ball.angle < 0)
        ball.angle += 2 * Math.PI;
    if (ball.angle > 2 * Math.PI)
        ball.angle -= 2 * Math.PI;
}

function checkCollision(oldX: number, oldY: number, ball: Ball, lPaddle: Paddle, rPaddle: Paddle) {
    let sign = 1;
    let posy = 0;
    if (ball.angle > 0.5 * Math.PI && ball.angle < 1.5 * Math.PI)
        sign = -1;
    if (sign === 1)
        posy = oldY + Math.tan(ball.angle) * (rPaddle.x - (0.5 * rPaddle.width) - oldX);
    else if (sign === -1)
        posy = oldY + Math.tan(ball.angle) * (lPaddle.x + (0.5 * lPaddle.width) - oldX);
    if (sign === 1 && posy >= rPaddle.y - 0.5 *  rPaddle.height && posy <= rPaddle.y + 0.5 * rPaddle.height)
        return (1);
    else if (sign === -1 && posy >= lPaddle.y - 0.5 * lPaddle.height && posy <= lPaddle.y + 0.5 * lPaddle.height)
        return (2);
    return (0);
}

function bounceAngle(ball: Ball, paddle: Paddle, side: string) {
    const ratio = (ball.y - paddle.y) / (paddle.height / 2);
    ball.speed = ball.ispeed + 0.5 * ball.ispeed * Math.abs(ratio);
    ball.angle = Math.PI * 0.25 * ratio;
    if (side === "right")
        ball.angle = Math.PI - ball.angle;
    norAngle(ball);
}

function moveBall(ball: Ball, lPaddle: Paddle, rPaddle: Paddle, lInput: keyInput, rInput: keyInput, game: gameState) {
    if (game.start) {
        let oldX = ball.x;
        let oldY = ball.y;
        let collision = 0;
        ball.x += Math.cos(ball.angle) * ball.speed;
        ball.y += Math.sin(ball.angle) * ball.speed;
        if ((ball.x > rPaddle.x - 0.5 * rPaddle.width && (ball.angle < 0.5 * Math.PI || ball.angle > 1.5 * Math.PI)) || (ball.x < lPaddle.x + 0.5 * lPaddle.width && (ball.angle > 0.5 * Math.PI && ball.angle < 1.5 * Math.PI)))
            collision = checkCollision(oldX, oldY, ball, lPaddle, rPaddle); // 0 = nothing || 1 = right || 2 = left
        if (collision === 1) {
            oldY = oldY + Math.tan(ball.angle) * (rPaddle.x - (0.5 * rPaddle.width) - oldX);
            oldX = rPaddle.x - (0.5 * rPaddle.width);
            bounceAngle(ball, rPaddle, "right");
            ball.x = oldX + Math.cos(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
            ball.y = oldY + Math.sin(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
        } else if (collision === 2) {
            oldY =  oldY - Math.tan(ball.angle) * (lPaddle.x + (0.5 * lPaddle.width) - oldX);
            oldX = lPaddle.x + (0.5 * lPaddle.width);
            bounceAngle(ball, lPaddle, "left");
            ball.x = oldX + Math.cos(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
            ball.y = oldY + Math.sin(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
        }
        if (ball.x > game.hazard.x - 37 && ball.x < game.hazard.x + 37) { // Hazard size is 50 but hitbox is 74 to cover ball radius
            if (ball.y > game.hazard.y - 37 && ball.y < game.hazard.y + 37) {
                hazardEffect(game, ball, lPaddle, rPaddle);
                game.hazard.type = "Default";
            }
        }
        if (ball.x > 1200) {
            lPaddle.score = String(Number(lPaddle.score) + 1);
            resetGame(ball, lPaddle, rPaddle, game, lInput, rInput);
        }
        if (ball.x < 0) {
            rPaddle.score = String(Number(rPaddle.score) + 1);
            resetGame(ball, lPaddle, rPaddle, game, lInput, rInput);
        }
        if (ball.y > 800) {
            ball.y = 800 - (ball.y - 800);
            ball.angle = 2 * Math.PI - ball.angle;
        } else if (ball.y < 0) {
            ball.y = -ball.y;
            ball.angle = 2 * Math.PI - ball.angle;
        }
        norAngle(ball);
    }
    setTimeout(() => moveBall(ball, lPaddle, rPaddle, lInput, rInput,  game), 10);
}

function movePaddleSolo(input: keyInput, lPaddle: Paddle, rPaddle: Paddle, game: gameState) {
    if (game.state === 1) {
        if (input.w)
            lPaddle.y -= lPaddle.speed;
        if (input.s)
            lPaddle.y += lPaddle.speed;
        if (input.arrowUp)
            rPaddle.y -= rPaddle.speed;
        if (input.arrowDown )
            rPaddle.y += rPaddle.speed;
        if (rPaddle.y < 0.5 * rPaddle.height)
            rPaddle.y = 0.5 * rPaddle.height;
        else if (rPaddle.y > 800 - rPaddle.height * 0.5)
            rPaddle.y = 800 - 0.5 * rPaddle.height;
        if (lPaddle.y < 0.5 * lPaddle.height)
            lPaddle.y = 0.5 * lPaddle.height;
        else if (lPaddle.y > 800 - lPaddle.height * 0.5)
            lPaddle.y = 800 - 0.5 * lPaddle.height;
    }
    setTimeout(() => movePaddleSolo(input, lPaddle, rPaddle, game), 10);
}

function movePaddle(lInput: keyInput, rInput: keyInput, lPaddle: Paddle, rPaddle: Paddle, game: gameState) {
    if (game.state === 1) {
        if (lInput.arrowUp || lInput.w)
            lPaddle.y -= lPaddle.speed;
        if (lInput.arrowDown || lInput.s)
            lPaddle.y += lPaddle.speed;
        if (rInput.arrowUp || rInput.w)
            rPaddle.y -= rPaddle.speed;
        if (rInput.arrowDown || rInput.s)
            rPaddle.y += rPaddle.speed;
        if (rPaddle.y < 0.5 * rPaddle.height)
            rPaddle.y = 0.5 * rPaddle.height;
        else if (rPaddle.y > 800 - rPaddle.height * 0.5)
            rPaddle.y = 800 - 0.5 * rPaddle.height;
        if (lPaddle.y < 0.5 * lPaddle.height)
            lPaddle.y = 0.5 * lPaddle.height;
        else if (lPaddle.y > 800 - lPaddle.height * 0.5)
            lPaddle.y = 800 - 0.5 * lPaddle.height;
    }
    setTimeout(() => movePaddle(lInput, rInput, lPaddle, rPaddle, game), 10);
}

function moveHazard(game: gameState, ball: Ball) {
    if (game.state === 1) {
        game.hazard.y += game.hazard.speed;
    }
    setTimeout(() => moveHazard(game, ball), 10);
}

function resetHazard(lPaddle: Paddle, rPaddle: Paddle, ball: Ball) {
    lPaddle.height = 200;
    rPaddle.height = 200;
    ball.ispeed = ball.ospeed;
}

function hazardEffect(game: gameState, ball: Ball, lPaddle: Paddle, rPaddle: Paddle) {
    let left = true;
    if (ball.angle > Math.PI * 0.5 && ball.angle < Math.PI * 1.5)
        left = false;
    switch (game.hazard.type) {
        case "BallSpeedUp":
            ball.ospeed = ball.ispeed;
            ball.speed *= 1.5;
            ball.ispeed *= 1.5;
            break;
        case "BarSizeUp":
            if (left)
                lPaddle.height += 100;
            else
                rPaddle.height += 100;
            break;
        case "BarSizeDown":
            if (left)
                lPaddle.height -= 50;
            else
                rPaddle.height -= 50;
            break;
        default:
            break;
    }
    setTimeout(() => resetHazard(lPaddle, rPaddle, ball), 5000);
}

function hazardGenerator(game: gameState) {
    if (game.start) {
        let type = Math.floor(Math.random() * 3);
        let rdm = Math.random();
        switch (type) {
            case 2 :
                game.hazard = new Hazard(450 + rdm * 300, 1 + Math.floor(rdm * 2), "BallSpeedUp");
                break;
            case 1:
                game.hazard = new Hazard(450 + rdm * 300, 1 + Math.floor(rdm * 2), "BarSizeUp");
                break;
            case 0:
                game.hazard = new Hazard(450 + rdm * 300, 1 + Math.floor(rdm * 2), "BarSizeDown");
                break;
            default:
                break;
        }
    }
    setTimeout(() => hazardGenerator(game), 10000);
}

function timerCheck(game: gameState) {
    if (game.state === 1) {
        if (game.timer.timeLeft === 0)
            game.start = true;
        else if (!game.timer.started)
            game.timer.start();
    }
    setTimeout(() => timerCheck(game), 1000);
}

function endSolo(intervalId: ReturnType<typeof setInterval>, game: gameState, solo: boolean) {
    if (!solo) {
        game.state = 2;
        clearInterval(intervalId);
        return ;
    }
    setTimeout(() => endSolo(intervalId, game, solo), 100);
}

function soloMode(player: Player, solo: boolean) {
    let ball = new Ball (1200 / 2, 800 / 2, 0, 8, 12, "#fcc800");
    let game = new gameState();
    let rPaddle = new Paddle(1170, 400, 20, 200, 10, "#fcc800");
    player.paddle.x = 30
    const intervalId = setInterval(() => {
        player.ws.send(JSON.stringify(player.paddle));
        player.ws.send(JSON.stringify(rPaddle));
        player.ws.send(JSON.stringify(ball));
        player.ws.send(JSON.stringify(game));
    }, 10);
    game.state = 1;
    moveBall(ball, player.paddle, rPaddle, player.input, player.input, game);
    movePaddleSolo(player.input, player.paddle, rPaddle, game);
    moveHazard(game, ball);
    hazardGenerator(game);
    timerCheck(game);
    endSolo(intervalId, game, solo);
}

// Netcode
function checkRoom(room: Room, game: gameState) {
    if (room.players.length !== 2) {
        game.state = 2;
        room.players.forEach(player => {
            player.ws.send(JSON.stringify({ type: "Disconnected" }));
            player.ws.close();
        });
        return ;
    }
    setTimeout(() => checkRoom(room, game), 100);
}

function leaveRoom(userId: number) {
    for (let i = 0 ; i < rooms.length; i++) {
        for (let j = 0; j < rooms[i].players.length; j++) {
            if (rooms[i].players[j].id === userId) {
                console.log("player: ", rooms[i].players[j].name, " with id: ", userId, " left room ", rooms[i].id);
                rooms[i].players.splice(j, 1);
                if (rooms[i].players.length === 0) {
                    console.log("room: ", rooms[i].id, " has been cleaned.");
                    rooms.splice(i, 1);
                }
                return ;
            }
        }
    }
    console.log("Player has not joined a room yet.");
}

function joinRoom(player: Player, roomId: number) {
    let id: number = -1;
    let i : number = 0;
    if (roomId !== -1) { // Joining a defined room (invite or tournaments)
        for (; i < rooms.length; i++) {
            if (rooms[i].id === roomId && rooms[i].players.length === 1) {
                player.paddle.x = 1200 - 30;
                rooms[i].players.push(player);
                break;
            }
        }
        if (id === -1) {
            player.paddle.x = 30;
            let room = new Room(roomId, player);
            rooms.push(room);
            return;
        }
    } else { // Basic random matchmaking
        for (; i < rooms.length; i++) {
            if (rooms[i].players.length === 1) {
                player.paddle.x = 1200 - 30;
                rooms[i].players.push(player);
                id = rooms[i].id;
                break;
            }
        }
        if (id === -1) {
            player.paddle.x = 30;
            let room = new Room(rooms.length, player);
            rooms.push(room);
            return;
        }
    }
    let ball = new Ball (1200 / 2, 800 / 2, 0, 8, 12, "#fcc800");
    let game = new gameState();
    const intervalId = setInterval(() => {
        let i = rooms.findIndex(room => room.id === id);
        if (i === -1) {
            clearInterval(intervalId);
            return;
        }
        if (rooms[i].players.length === 2) {
            rooms[i].players[0].ws.send(JSON.stringify(rooms[i].players[0].paddle));
            rooms[i].players[0].ws.send(JSON.stringify(rooms[i].players[1].paddle));
            rooms[i].players[0].ws.send(JSON.stringify(ball));
            rooms[i].players[0].ws.send(JSON.stringify(game));
            rooms[i].players[1].ws.send(JSON.stringify(rooms[i].players[0].paddle));
            rooms[i].players[1].ws.send(JSON.stringify(rooms[i].players[1].paddle));
            rooms[i].players[1].ws.send(JSON.stringify(ball));
            rooms[i].players[1].ws.send(JSON.stringify(game));
        } else
            clearInterval(intervalId);
    }, 10);
    game.state = 1;
    moveBall(ball, rooms[i].players[0].paddle, rooms[i].players[1].paddle, rooms[i].players[0].input, rooms[i].players[1].input, game);
    movePaddle(rooms[i].players[0].input, rooms[i].players[1].input, rooms[i].players[0].paddle, rooms[i].players[1].paddle, game);
    moveHazard(game, ball);
    hazardGenerator(game);
    timerCheck(game);
    checkRoom(rooms[i], intervalId, game);
}

const fastify = Fastify();

fastify.register(fastifyWebsocket);

fastify.register(async function (fastify) {
    fastify.get('/ws', { websocket: true }, (socket, req) => {
        console.log("Client connected");
        const userId = generateId();
        const player = new Player(userId, socket);
        let init = false;
        let room = -1;
        let solo: boolean;
        socket.on("message", (message) => {
            const msg = JSON.parse(message.toString());
            if (!init && msg.type === "socketInit") {
                const {data, success, error} = initSchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                player.name = data.nick;
                if (data.room)
                    room = msg.room;
                init = true;
            } else if (msg.type === "input") {
                const {data, success, error} = inputSchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                resetInput(player.input);
                inputHandler(data.key, data.state, player.input);
            } else if (init && msg.type === "ready") {
                const {data, success, error} = readySchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.error(error);
                    return ;
                }
                console.log(data.mode);
                if (data.mode === "remote")
                    joinRoom(player, room);
                else if (data.mode === "local") {
                    solo = true;
                    soloMode(player, solo);
                }
            }
        });
        socket.on("close", () => {
            leaveRoom(userId);
            solo = false;
            console.log("Client disconnected");
        });

    });
});

fastify.listen({ port: 4443, host: '0.0.0.0' }, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
});