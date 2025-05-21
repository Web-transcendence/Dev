//Solo mode
import {
    Ball,
    bounceAngle,
    checkCollision,
    gameState,
    hazardEffect, hazardGenerator,
    keyInput, moveHazard,
    norAngle,
    Paddle,
    Player, resetHazard, resetInput,
    Timer, timerCheck
} from "./api.js";
import {rooms} from "./netcode.js";

function resetGameSolo(ball: Ball, lPaddle: Paddle, rPaddle: Paddle, game: gameState, lInput: keyInput, rInput: keyInput) {
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
    resetInput(lInput, "left");
    resetInput(rInput, "right");
}

function moveBallSolo(ball: Ball, lPaddle: Paddle, rPaddle: Paddle, lInput: keyInput, rInput: keyInput, game: gameState) {
    if (game.start) {
        let oldX = ball.x;
        let oldY = ball.y;
        let collision = 0;
        ball.x += Math.cos(ball.angle) * ball.speed;
        ball.y += Math.sin(ball.angle) * ball.speed;
        if ((ball.x > rPaddle.x - 0.5 * rPaddle.width && (ball.angle < 0.5 * Math.PI || ball.angle > 1.5 * Math.PI)) || (ball.x < lPaddle.x + 0.5 * lPaddle.width && (ball.angle > 0.5 * Math.PI && ball.angle < 1.5 * Math.PI)))
            collision = checkCollision(oldX, oldY, ball, lPaddle, rPaddle); // 0 = nothing || 1 = left || 2 = right
        if (collision === 1) {
            oldY = oldY + Math.tan(ball.angle) * (lPaddle.x + (0.5 * lPaddle.width) - oldX);
            oldX = lPaddle.x + (0.5 * lPaddle.width);
            bounceAngle(ball, lPaddle, "left");
            ball.x = oldX + Math.cos(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
            ball.y = oldY + Math.sin(ball.angle) * (Math.sqrt(Math.pow(ball.y - oldY, 2) + Math.pow(ball.x - oldX, 2)));
        } else if (collision === 2) {
            oldY = oldY - Math.tan(ball.angle) * (rPaddle.x - (0.5 * rPaddle.width) - oldX);
            oldX = rPaddle.x - (0.5 * lPaddle.width);
            bounceAngle(ball, rPaddle, "right");
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
            resetGameSolo(ball, lPaddle, rPaddle, game, lInput, rInput);
        }
        if (ball.x < 0) {
            rPaddle.score = String(Number(rPaddle.score) + 1);
            resetGameSolo(ball, lPaddle, rPaddle, game, lInput, rInput);
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
    setTimeout(() => moveBallSolo(ball, lPaddle, rPaddle, lInput, rInput,  game), 10);
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

function endSolo(intervalId: ReturnType<typeof setInterval>, game: gameState, solo: boolean) {
    if (!solo) {
        game.state = 2;
        clearInterval(intervalId);
        return ;
    }
    setTimeout(() => endSolo(intervalId, game, solo), 100);
}

export function soloMode(player: Player, solo: boolean) {
    let ball = new Ball (1200 / 2, 800 / 2, 0, 8, 12, "#fcc800");
    let game = new gameState();
    let rPaddle = new Paddle(1170, 400, 20, 200, 10, "#fcc800");
    player.paddle.x = 30
    const intervalId = setInterval(() => {
        const payload = {
            type: "gameUpdate",
            paddle1: player.paddle,
            paddle2: rPaddle,
            ball: ball,
            game: game
        };
        player.ws.send(JSON.stringify(payload));
    }, 10);
    game.state = 1;
    moveBallSolo(ball, player.paddle, rPaddle, player.input, player.input, game);
    movePaddleSolo(player.input, player.paddle, rPaddle, game);
    moveHazard(game, ball);
    hazardGenerator(game);
    timerCheck(game);
    endSolo(intervalId, game, solo);
}