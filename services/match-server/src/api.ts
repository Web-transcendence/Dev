import Fastify from 'fastify';
import fastifyWebsocket from '@fastify/websocket';
import { WebSocket } from "ws";
import { z } from "zod";


const inputSchema = z.object({ state: z.string(), key: z.string() });

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
    ws: WebSocket;
    paddle: Paddle = new Paddle(0, 400, 20, 200, 10, "#fcc800");
    input: keyInput = new keyInput();
    game: gameState = new gameState();
    constructor(ws: WebSocket) {
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

function inputHandler(key: string, state: string, input: keyInput, game: gameState) {
    let upordown = false;
    if (game.state === 1) {
        if (state === "down")
            upordown = true;
        if (key === "w")
            input.w = upordown;
        if (key === "s")
            input.s = upordown;
        if (key === "ArrowUp")
            input.arrowUp = upordown;
        if (key === "ArrowDown")
            input.arrowDown = upordown;
    }
    else if (state === "down")
        game.state = 1;
}

function resetInput(lInput: keyInput, rInput: keyInput) {
    rInput.arrowUp = false;
    rInput.arrowDown = false;
    rInput.w = false;
    rInput.arrowUp = false;
    lInput.arrowUp = false;
    lInput.arrowDown = false;
    lInput.w = false;
    lInput.arrowUp = false;
}

function resetGame(ball: Ball, lPaddle: Paddle, rPaddle: Paddle, lGame: gameState, rGame: gameState, lInput: keyInput, rInput: keyInput) {
    lGame.start = false;
    rGame.start = false;
    lGame.timer = new Timer (0, 2);
    rGame.timer = new Timer (0, 2);
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
    lGame.hazard.type = "Default";
    rGame.hazard.type = "Default";
    if (rPaddle.score === lGame.maxScore || lPaddle.score === lGame.maxScore) {
        lGame.state = 2;
        lGame.score1 = lPaddle.score;
        lGame.score2 = rPaddle.score;
        rGame.state = 2;
        rGame.score1 = lPaddle.score;
        rGame.score2 = rPaddle.score;
        rPaddle.score = "0";
        lPaddle.score = "0";
    }
    resetInput(lInput, rInput);
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

function moveBall(ball: Ball, lPaddle: Paddle, rPaddle: Paddle, lInput: keyInput, rInput: keyInput, lGame: gameState, rGame: gameState) {
    if (lGame.start && rGame.start) {
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
        if (ball.x > lGame.hazard.x - 37 && ball.x < lGame.hazard.x + 37) { // Hazard size is 50 but hitbox is 74 to cover ball radius
            if (ball.y > lGame.hazard.y - 37 && ball.y < lGame.hazard.y + 37) {
                hazardEffect(lGame, rGame, ball, lPaddle, rPaddle);
                lGame.hazard.type = "Default";
                rGame.hazard.type = "Default";
            }
        }
        if (ball.x > 1200) {
            lPaddle.score = String(Number(lPaddle.score) + 1);
            resetGame(ball, lPaddle, rPaddle, lGame, rGame, lInput, rInput);
        }
        if (ball.x < 0) {
            rPaddle.score = String(Number(rPaddle.score) + 1);
            resetGame(ball, lPaddle, rPaddle, lGame, rGame, lInput, rInput);
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
    setTimeout(() => moveBall(ball, lPaddle, rPaddle, lInput, rInput,  lGame, rGame), 10);
}

function movePaddle(lInput: keyInput, rInput: keyInput, lPaddle: Paddle, rPaddle: Paddle, lGame: gameState, rGame: gameState) {
    if (lGame.state === 1 && rGame.state === 1) {
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
    setTimeout(() => movePaddle(lInput, rInput, lPaddle, rPaddle, lGame, rGame), 10);
}

function moveHazard(lGame: gameState, rGame: gameState, ball: Ball) {
    if (lGame.state === 1 && rGame.state === 1) {
        lGame.hazard.y += lGame.hazard.speed;
        rGame.hazard.y += rGame.hazard.speed;
    }
    setTimeout(() => moveHazard(lGame, rGame, ball), 10);
}

function resetHazard(lPaddle: Paddle, rPaddle: Paddle, ball: Ball) {
    lPaddle.height = 200;
    rPaddle.height = 200;
    ball.ispeed = ball.ospeed;
}

function hazardEffect(lGame: gameState, rGame: gameState, ball: Ball, lPaddle: Paddle, rPaddle: Paddle) {
    let left = true;
    if (ball.angle > Math.PI * 0.5 && ball.angle < Math.PI * 1.5)
        left = false;
    switch (lGame.hazard.type) {
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

function hazardGenerator(lGame: gameState, rGame: gameState) {
    if (lGame.start && rGame.start) {
        let type = Math.floor(Math.random() * 3);
        let rdm = Math.random();
        switch (type) {
            case 2 :
                lGame.hazard = new Hazard(450 + rdm * 300, 1 + Math.floor(rdm * 2), "BallSpeedUp");
                rGame.hazard = new Hazard(450 + rdm * 300, 1 + Math.floor(rdm * 2), "BallSpeedUp");
                break;
            case 1:
                lGame.hazard = new Hazard(450 + rdm * 300, 1 + Math.floor(rdm * 2), "BarSizeUp");
                rGame.hazard = new Hazard(450 + rdm * 300, 1 + Math.floor(rdm * 2), "BarSizeUp");
                break;
            case 0:
                lGame.hazard = new Hazard(450 + rdm * 300, 1 + Math.floor(rdm * 2), "BarSizeDown");
                rGame.hazard = new Hazard(450 + rdm * 300, 1 + Math.floor(rdm * 2), "BarSizeDown");
                break;
            default:
                break;
        }
    }
    setTimeout(() => hazardGenerator(lGame, rGame), 10000);
}

function timerCheck(lGame: gameState, rGame: gameState) {
    if (lGame.state === 1 && rGame.state === 1) {
        if (lGame.timer.timeLeft === 0 && rGame.timer.timeLeft === 0) {
            lGame.start = true;
            rGame.start = true;
        }
        else if (!lGame.timer.started && !rGame.timer.started) {
            lGame.timer.start();
            rGame.timer.start();
        }

    }
    setTimeout(() => timerCheck(lGame, rGame), 1000);
}

function joinRoom(player: Player) {
    let id: number = -1;
    let i : number = 0;
    for (; i < rooms.length; i++) {
        if (rooms[i].players.length === 1) {
            player.paddle.x = 1200 - 30;
            rooms[i].players.push(player);
            id = rooms[i].id;
            break ;
        }
    }
    if (id === -1) {
        player.paddle.x = 30;
        let room = new Room(rooms.length, player);
        rooms.push(room);
        return ;
    }
    let ball = new Ball (1200 / 2, 800 / 2, 0, 8, 12, "#fcc800");
    const intervalId = setInterval(() => {
        let i = rooms.findIndex(room => room.id === id);
        if (i === -1) {
            clearInterval(intervalId);
            return;
        }
        rooms[i].players[0].ws.send(JSON.stringify(rooms[i].players[0].paddle));
        rooms[i].players[0].ws.send(JSON.stringify(rooms[i].players[1].paddle));
        rooms[i].players[0].ws.send(JSON.stringify(ball));
        rooms[i].players[0].ws.send(JSON.stringify(rooms[i].players[0].game));
        rooms[i].players[1].ws.send(JSON.stringify(rooms[i].players[0].paddle));
        rooms[i].players[1].ws.send(JSON.stringify(rooms[i].players[1].paddle));
        rooms[i].players[1].ws.send(JSON.stringify(ball));
        rooms[i].players[1].ws.send(JSON.stringify(rooms[i].players[1].game));
    }, 10);
    moveBall(ball, rooms[i].players[0].paddle, rooms[i].players[1].paddle, rooms[i].players[0].input, rooms[i].players[1].input,  rooms[i].players[0].game, rooms[i].players[1].game);
    movePaddle(rooms[i].players[0].input, rooms[i].players[1].input, rooms[i].players[0].paddle, rooms[i].players[1].paddle, rooms[i].players[0].game, rooms[i].players[1].game);
    moveHazard(rooms[i].players[0].game, rooms[i].players[1].game, ball);
    hazardGenerator(rooms[i].players[0].game, rooms[i].players[1].game);
    timerCheck(rooms[i].players[0].game, rooms[i].players[1].game);
}

const fastify = Fastify();

fastify.register(fastifyWebsocket);

fastify.register(async function (fastify) {
    fastify.get('/ws', { websocket: true }, (socket, req) => {
        console.log("Client connected");
        let player = new Player(socket);
        socket.on("message", (message) => {
            const {data, success, error} = inputSchema.safeParse(JSON.parse(message.toString()));
            if (!success || !data) {
                console.error(error);
                return ;
            }
            inputHandler(data.key, data.state, player.input, player.game);
        });
        socket.on("close", () => {
            console.log("Client disconnected");
        });
        joinRoom(player);
    });
});

fastify.listen({ port: 4443, host: '0.0.0.0' }, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
});