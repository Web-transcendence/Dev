import Fastify from 'fastify';
import fastifyWebsocket from '@fastify/websocket';
import { WebSocket } from "ws";
import { z } from "zod";
import { insertMatchResult, getMatchHistory } from "./database.js";
import pongRoutes from "./routes.js";

export const inputSchema = z.object({ state: z.string(), key: z.string() });
export const initSchema = z.object({ nick: z.string(), room: z.number() });
export const readySchema = z.object({ mode: z.string() });

export class Ball {
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

export class Paddle {
    name: string = "Player 2";
    x: number;
    y: number;
    width: number;
    height: number;
    speed: number;
    color: string;
    score: string = "0";
    constructor(x: number, y: number, width: number, height: number, speed: number, color: string) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.speed = speed;
        this.color = color;
    }
    toJSON() {
        return {type: "Paddle", name: this.name, x: this.x, y: this.y, width: this.width, height: this.height, color: this.color, score: this.score};
    }
}

export class keyInput {
    arrowUp: boolean = false;
    arrowDown: boolean = false;
    w: boolean = false;
    s: boolean = false;
}

export class Hazard {
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

export class gameState {
    state: number = 0;
    start: boolean = false;
    maxScore: string = "6";
    score1: string = "0";
    score2: string = "0";
    winner: string = "none";
    hazard: Hazard = new Hazard(0, 0, "Default");
    timer: Timer = new Timer (0, 3);
    toJSON() {
        return {type: "Game", state: this.state, start: this.start, score1: this.score1, score2: this.score2, hazard: this.hazard, timer: this.timer, winner: this.winner};
    }
}

export class Room {
    id: number;
    players: Player[] = [];
    ended = false;
    constructor (id: number) {
        this.id = id;
    }
}

export class Player {
    name: string = "Default";
    dbId: number = -1;
    id: number;
    ws: WebSocket;
    frequency: number = 10;
    paddle: Paddle = new Paddle(0, 400, 20, 200, 10, "#fcc800");
    input: keyInput = new keyInput();
    constructor(id: number, ws: WebSocket) {
        this.id = id;
        this.ws = ws;
    }
}

export class Timer {
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

const ids = new Set<number>();

export function generateId() {
    let newId: number;
    do {
        newId = Math.floor(Math.random() * 10000);
    } while (ids.has(newId));
    ids.add(newId);
    return (newId);
}

export function inputHandler(key: string, state: string, input: keyInput) {
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

export function resetInput(Input: keyInput) {
    Input.arrowUp = false;
    Input.arrowDown = false;
    Input.w = false;
    Input.arrowUp = false;
}

export function resetGame(ball: Ball, player1: Player, player2: Player, game: gameState) {
    game.start = false;
    game.timer = new Timer (0, 2);
    if (ball.x < 0)
        ball.angle = Math.PI;
    else
        ball.angle = 0;
    ball.x = 0.5 * 1200;
    ball.y = 0.5 * 800;
    resetHazard(player1.paddle, player2.paddle, ball)
    ball.speed = ball.ispeed;
    player1.paddle.y = 0.5 * 800;
    player2.paddle.y = 0.5 * 800;
    game.hazard.type = "Default";
    if (player1.paddle.score === game.maxScore || player2.paddle.score === game.maxScore) {
        let winner = 2;
        if (player1.paddle.score > player2.paddle.score)
            winner = 0;
        else if (player1.paddle.score < player2.paddle.score)
            winner = 1;
        insertMatchResult(player1.dbId, player2.dbId, Number(player1.paddle.score), Number(player2.paddle.score), winner);
        game.state = 2;
        game.score1 = player1.paddle.score;
        game.score2 = player2.paddle.score;
        if (player1.paddle.score > player2.paddle.score)
            game.winner = player1.name;
        else
            game.winner = player2.name;
        player1.paddle.score = "0";
        player2.paddle.score = "0";
        console.log(getMatchHistory(player1.dbId));
    }
    resetInput(player1.input);
    resetInput(player2.input);
}

export function norAngle(ball: Ball) {
    if (ball.angle < 0)
        ball.angle += 2 * Math.PI;
    if (ball.angle > 2 * Math.PI)
        ball.angle -= 2 * Math.PI;
}

export function checkCollision(oldX: number, oldY: number, ball: Ball, lPaddle: Paddle, rPaddle: Paddle) {
    let sign = 1;
    let posy = 0;
    if (ball.angle > 0.5 * Math.PI && ball.angle < 1.5 * Math.PI)
        sign = -1;
    if (sign === 1)
        posy = oldY + Math.tan(ball.angle) * (rPaddle.x - (0.5 * rPaddle.width) - oldX);
    else if (sign === -1)
        posy = oldY + Math.tan(ball.angle) * (lPaddle.x + (0.5 * lPaddle.width) - oldX);
    if (sign === 1 && posy >= rPaddle.y - 0.5 *  rPaddle.height && posy <= rPaddle.y + 0.5 * rPaddle.height)
        return (2);
    else if (sign === -1 && posy >= lPaddle.y - 0.5 * lPaddle.height && posy <= lPaddle.y + 0.5 * lPaddle.height)
        return (1);
    return (0);
}

export function bounceAngle(ball: Ball, paddle: Paddle, side: string) {
    const ratio = (ball.y - paddle.y) / (paddle.height / 2);
    ball.speed = ball.ispeed + 0.5 * ball.ispeed * Math.abs(ratio);
    ball.angle = Math.PI * 0.25 * ratio;
    if (side === "right")
        ball.angle = Math.PI - ball.angle;
    norAngle(ball);
}

export function moveBall(ball: Ball, player1: Player, player2: Player, game: gameState) {
    if (game.start) {
        let oldX = ball.x;
        let oldY = ball.y;
        let collision = 0;
        ball.x += Math.cos(ball.angle) * ball.speed;
        ball.y += Math.sin(ball.angle) * ball.speed;
        if ((ball.x > player2.paddle.x - 0.5 * player2.paddle.width && (ball.angle < 0.5 * Math.PI || ball.angle > 1.5 * Math.PI)) || (ball.x < player1.paddle.x + 0.5 * player1.paddle.width && (ball.angle > 0.5 * Math.PI && ball.angle < 1.5 * Math.PI)))
            collision = checkCollision(oldX, oldY, ball, player1.paddle, player2.paddle); // 0 = nothing || 1 = left || 2 = right
        if (collision === 1) {
            oldY = oldY - Math.tan(ball.angle) * (player1.paddle.x + (0.5 * player1.paddle.width) - oldX);
            oldX = player1.paddle.x + (0.5 * player1.paddle.width);
            bounceAngle(ball, player1.paddle, "left");
            ball.x = oldX + Math.cos(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
            ball.y = oldY + Math.sin(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
        } else if (collision === 2) {
            oldY =  oldY + Math.tan(ball.angle) * (player2.paddle.x - (0.5 * player2.paddle.width) - oldX);
            oldX = player2.paddle.x - (0.5 * player2.paddle.width);
            bounceAngle(ball, player2.paddle, "right");
            ball.x = oldX + Math.cos(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
            ball.y = oldY + Math.sin(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
        }
        if (ball.x > game.hazard.x - 37 && ball.x < game.hazard.x + 37) { // Hazard size is 50 but hitbox is 74 to cover ball radius
            if (ball.y > game.hazard.y - 37 && ball.y < game.hazard.y + 37) {
                hazardEffect(game, ball, player1.paddle, player2.paddle);
                game.hazard.type = "Default";
            }
        }
        if (ball.x > 1200) {
            player1.paddle.score = String(Number(player1.paddle.score) + 1);
            resetGame(ball, player1, player2, game);
        }
        if (ball.x < 0) {
            player2.paddle.score = String(Number(player2.paddle.score) + 1);
            resetGame(ball, player1, player2, game);
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
    setTimeout(() => moveBall(ball, player1, player2, game), 10);
}

export function movePaddle(lInput: keyInput, rInput: keyInput, lPaddle: Paddle, rPaddle: Paddle, game: gameState) {
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

export function moveHazard(game: gameState, ball: Ball) {
    if (game.state === 1) {
        game.hazard.y += game.hazard.speed;
    }
    setTimeout(() => moveHazard(game, ball), 10);
}

export function resetHazard(lPaddle: Paddle, rPaddle: Paddle, ball: Ball) {
    lPaddle.height = 200;
    rPaddle.height = 200;
    ball.ispeed = ball.ospeed;
}

export function hazardEffect(game: gameState, ball: Ball, lPaddle: Paddle, rPaddle: Paddle) {
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

export function hazardGenerator(game: gameState) {
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

export function timerCheck(game: gameState) {
    if (game.state === 1) {
        if (game.timer.timeLeft === 0)
            game.start = true;
        else if (!game.timer.started)
            game.timer.start();
    }
    setTimeout(() => timerCheck(game), 1000);
}

export const INTERNAL_PASSWORD = process.env?.SECRET_KEY;

const fastify = Fastify();

fastify.register(fastifyWebsocket);

fastify.register(pongRoutes);

fastify.listen({ port: 4443, host: '0.0.0.0' }, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
});