import Fastify from 'fastify';
import fastifyWebsocket from '@fastify/websocket';
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
    toJSON() {
        return {type: "Game", state: this.state, start: this.start, score1: this.score1, score2: this.score2, hazard: this.hazard};
    }
}

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
        if (key === "Control")
            game.start = true;
    }
    else if (state === "down")
        game.state = 1;
}

function resetInput(input: keyInput) {
    input.arrowUp = false;
    input.arrowDown = false;
    input.w = false;
    input.arrowUp = false;
}

function resetGame(ball: Ball, lPaddle: Paddle, rPaddle: Paddle, game: gameState, input: keyInput) {
    game.start = false;
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
    resetInput(input);
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

function moveBall(ball: Ball, lPaddle: Paddle, rPaddle: Paddle, input: keyInput, game: gameState) {
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
            resetGame(ball, lPaddle, rPaddle, game, input);
        }
        if (ball.x < 0) {
            rPaddle.score = String(Number(rPaddle.score) + 1);
            resetGame(ball, lPaddle, rPaddle, game, input);
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
    setTimeout(() => moveBall(ball, lPaddle, rPaddle, input, game), 10);
}

function movePaddle(input: keyInput, lPaddle: Paddle, rPaddle: Paddle, game: gameState) {
    if (game.state === 1) {
        if (input.arrowUp)
            rPaddle.y -= rPaddle.speed;
        if (input.arrowDown)
            rPaddle.y += rPaddle.speed;
        if (input.w)
            lPaddle.y -= lPaddle.speed;
        if (input.s)
            lPaddle.y += lPaddle.speed;
        if (rPaddle.y < 0.5 * rPaddle.height)
            rPaddle.y = 0.5 * rPaddle.height;
        else if (rPaddle.y > 800 - rPaddle.height * 0.5)
            rPaddle.y = 800 - 0.5 * rPaddle.height;
        if (lPaddle.y < 0.5 * lPaddle.height)
            lPaddle.y = 0.5 * lPaddle.height;
        else if (lPaddle.y > 800 - lPaddle.height * 0.5)
            lPaddle.y = 800 - 0.5 * lPaddle.height;
    }
    setTimeout(() => movePaddle(input, lPaddle, rPaddle, game), 10);
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
        switch (type) {
            case 2 :
                game.hazard = new Hazard(450 + Math.random() * 300, 1 + Math.floor(Math.random() * 2), "BallSpeedUp");
                break;
            case 1:
                game.hazard = new Hazard(450 + Math.random() * 300, 1 + Math.floor(Math.random() * 2), "BarSizeUp");
                break;
            case 0:
                game.hazard = new Hazard(450 + Math.random() * 300, 1 + Math.floor(Math.random() * 2), "BarSizeDown");
                break;
            default:
                break;
        }
    }
    setTimeout(() => hazardGenerator(game), 10000);
}

const fastify = Fastify({
    logger: true
});

fastify.register(fastifyWebsocket);

fastify.register(async function (fastify) {
    fastify.get('/ws', { websocket: true }, (socket) => {
        console.log("Client connected");
        let ball = new Ball (1200 / 2, 800 / 2, 0, 8, 12, "#fcc800");
        let lPaddle = new Paddle(30, 800 / 2, 20, 200, 10, "#fcc800");
        let rPaddle = new Paddle(1200 - 30, 800 / 2, 20, 200, 10, "#fcc800");
        let input = new keyInput();
        let game = new gameState();

        socket.on("message", (message) => {
            const {data, success, error} = inputSchema.safeParse(JSON.parse(message.toString()));
            if (!success || !data) {
                console.error(error);
                return ;
            }
            inputHandler(data.key, data.state, input, game);
        });

        const intervalId = setInterval(() => {
            socket.send(JSON.stringify(lPaddle));
            socket.send(JSON.stringify(rPaddle));
            socket.send(JSON.stringify(ball));
            socket.send(JSON.stringify(game));
        }, 10);

        socket.on("close", () => {
            clearInterval(intervalId);
            console.log("Client disconnected");
        });

        moveBall(ball, lPaddle, rPaddle, input, game);
        movePaddle(input, lPaddle, rPaddle, game);
        moveHazard(game, ball);
        hazardGenerator(game);
    });
});

fastify.listen({ port: 4443, host: '0.0.0.0' }, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`);
});