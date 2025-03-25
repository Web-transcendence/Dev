const canvas = document.getElementById("gameCanvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
let connect: boolean = true;

class Ball {
    x: number;
    y: number;
    radius: number;
    color: string;
    constructor(x: number, y: number, radius: number, color: string) {
        this.x = x;
        this.y = y;
        this.radius = radius;
        this.color = color;
    }
}

class Paddle {
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
    score: string;
    constructor(x: number, y: number, width: number, height: number, color: string) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = color;
        this.score = "0";
    }
}

class gameState {
    state: number = 0;
    start: boolean = false;
    score1: string = "0";
    score2: string = "0";
    hazard: Hazard = new Hazard (0, 0, "Default");
}

class Hazard {
    x: number;
    y: number;
    type: string;
    constructor (x: number, y: number, type: string) {
        this.x = x;
        this.y = y;
        this.type = type;
    }
}

class Assets {
    BarUp: HTMLImageElement = new Image();
    BarDown: HTMLImageElement = new Image();
    BallUp: HTMLImageElement = new Image();
    constructor () {
        this.BarUp.src = "./assets/barup.png";
        this.BarDown.src = "./assets/bardown.png";
        this.BallUp.src = "./assets/ballup.png";
    }
}

let animFrame = 0;
let animLoop = 1;
let game = new gameState();
let ball = new Ball(canvas.width / 2, canvas.height / 2, 10, "#fcc800");
let lPaddle = new Paddle(30, canvas.height / 2, 20, 200, "#fcc800");
let rPaddle = new Paddle(canvas.width - 30, canvas.height / 2, 20, 200, "#fcc800");
let asset = new Assets;

function titleScreen() {
    ctx.fillStyle = "#364153";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#101828";
    ctx.fillRect(15, 15, canvas.width - 30, canvas.height - 30);
    ctx.fillStyle = "#fcc800";
    ctx.font = "84px 'Press Start 2P'";
    ctx.textAlign = "center"
    ctx.fillText("Pong Game", canvas.width * 0.5, canvas.height * 0.5);
    if (animFrame === 0 || animFrame === 1)
        ctx.fillStyle = "#fcc800";
    else if (animFrame === 2 || animFrame === 3)
        ctx.fillStyle = "#ffd014";
    else if (animFrame === 4 || animFrame === 5)
        ctx.fillStyle = "#ffd52b";
    else if (animFrame === 6 || animFrame === 7)
        ctx.fillStyle = "#ffd83e";
    else if (animFrame === 8 || animFrame === 9)
        ctx.fillStyle = "#ffdb5e";
    ctx.font = "30px 'Press Start 2P'";
    ctx.fillText("Press any key", canvas.width * 0.5, canvas.height * 0.5 + 60 + animFrame);
    animFrame += animLoop;
    if (animFrame === 0 || animFrame === 9)
        animLoop *= -1;
}

function endScreen() {
    ctx.fillStyle = "#101828";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#364153";
    for (let i = 0; i < canvas.height; i += 60) {
        ctx.fillRect(canvas.width * 0.5 - 4, i, 8, 30);
    }
    ctx.fillStyle = ball.color;
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = lPaddle.color;
    ctx.fillRect(lPaddle.x - lPaddle.width * 0.5, lPaddle.y - lPaddle.height * 0.5, lPaddle.width, lPaddle.height);
    ctx.fillStyle = rPaddle.color;
    ctx.fillRect(rPaddle.x - rPaddle.width * 0.5, rPaddle.y - rPaddle.height * 0.5, rPaddle.width, rPaddle.height);
    ctx.fillStyle = "#fcc800";
    ctx.font = "48px 'Press Start 2P'";
    ctx.textAlign = "left"
    ctx.fillText(game.score2, canvas.width * 0.5 + 46, 80);
    ctx.textAlign = "right"
    ctx.fillText(game.score1, canvas.width * 0.5 - 40, 80);
    ctx.fillStyle = "#ddae00";
    ctx.font = "60px 'Press Start 2P'";
    ctx.textAlign = "center";
    if (game.score1 > game.score2)
        ctx.fillText("Player 1 Wins", canvas.width * 0.5, canvas.height * 0.4);
    else
        ctx.fillText("Player 2 Wins", canvas.width * 0.5, canvas.height * 0.4);
    ctx.font = "26px 'Press Start 2P'";
    ctx.textAlign = "center";
    ctx.fillText("Press any key to restart game", canvas.width * 0.5, canvas.height * 0.65);
}

function drawHazard() {
    switch (game.hazard.type) {
        case "BarSizeUp":
            ctx.drawImage(asset.BarUp, game.hazard.x - 25, game.hazard.y - 25, 50, 50);
            //ctx.fillStyle = "green";
            //ctx.fillRect(game.hazard.x - 25, game.hazard.y - 25, 50, 50);
            break;
        case "BarSizeDown":
            ctx.drawImage(asset.BarDown, game.hazard.x - 25, game.hazard.y - 25, 50, 50);
            //ctx.fillStyle = "red";
            //ctx.fillRect(game.hazard.x - 25, game.hazard.y - 25, 50, 50);
            break;
        case "BallSpeedUp":
            ctx.drawImage(asset.BallUp, game.hazard.x - 25, game.hazard.y - 25, 50, 50);
            //ctx.fillStyle = "blue";
            //ctx.fillRect(game.hazard.x - 25, game.hazard.y - 25, 50, 50);
            break;
        default:
            break;
    }
}

function mainLoop() {
    // Background
    ctx.fillStyle = "#101828";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Mid
    ctx.fillStyle = "#364153";
    for (let i = 0; i < canvas.height; i += 60)
        ctx.fillRect(canvas.width * 0.5 - 4, i, 8, 30);
    // Score
    ctx.fillStyle = "#fcc800";
    ctx.font = "48px 'Press Start 2P'";
    ctx.textAlign = "left"
    ctx.fillText(rPaddle.score, canvas.width * 0.5 + 46, 80);
    ctx.textAlign = "right"
    ctx.fillText(lPaddle.score, canvas.width * 0.5 - 40, 80);
    // Hazard
    drawHazard();
    // Ball
    ctx.fillStyle = ball.color;
    ctx.fillRect(ball.x - ball.radius, ball.y - ball.radius + 4, ball.radius * 2, (ball.radius - 4) * 2)
    ctx.fillRect(ball.x - ball.radius + 4, ball.y - ball.radius, (ball.radius - 4) * 2, ball.radius * 2)
    // lPaddle
    ctx.fillStyle = lPaddle.color;
    ctx.fillRect(lPaddle.x + 3 - lPaddle.width * 0.5, lPaddle.y - lPaddle.height * 0.5, lPaddle.width - 6, lPaddle.height);
    ctx.fillRect(lPaddle.x - lPaddle.width * 0.5, lPaddle.y + 3 - lPaddle.height * 0.5, lPaddle.width, lPaddle.height - 6);
    // rPaddle
    ctx.fillStyle = rPaddle.color;
    ctx.fillRect(rPaddle.x + 3 - rPaddle.width * 0.5, rPaddle.y - rPaddle.height * 0.5, rPaddle.width - 6, rPaddle.height);
    ctx.fillRect(rPaddle.x - rPaddle.width * 0.5, rPaddle.y + 3 - rPaddle.height * 0.5, rPaddle.width, rPaddle.height - 6);
}

function gameLoop () {
    switch (game.state) {
        case 0:
            titleScreen();
            break;
        case 1:
            mainLoop();
            break;
        case 2:
            endScreen();
            break;
    }
    if (connect)
        requestAnimationFrame(gameLoop);
}

gameLoop();
try {
    const socket = new WebSocket("http://localhost:4443/ws");
    socket.addEventListener("open", (event) => {
        console.log("Connected");
    })
    socket.onopen = (() => {
        socket.onerror = (err => {
            console.error(err);
        })
        window.addEventListener("keydown", (event) => {
            socket.send(JSON.stringify({ type: "input", key: event.key, state: "down" }));
        });
        window.addEventListener("keyup", (event) => {
            socket.send(JSON.stringify({ type: "input", key: event.key, state: "up" }));
        });
        socket.onopen = function () { return console.log("Connected to server"); };
        socket.onmessage = function (event) {
            const data = JSON.parse(event.data);
            const existingScript = document.querySelector('script[src="/static/dist/pong.js"]');
            if (!existingScript) {
                socket.close();
                connect = false;
                console.log("Pong script finished, socket closed.");
            }
            switch (data.type) {
                case "Game":
                    game.state = data.state;
                    game.score1 = data.score1;
                    game.score2 = data.score2;
                    game.start = data.start;
                    game.hazard.x = data.hazard.x;
                    game.hazard.y = data.hazard.y;
                    game.hazard.type = data.hazard.type;
                    break;
                case "Ball":
                    ball.x = data.x;
                    ball.y = data.y;
                    ball.radius = data.radius;
                    ball.color = data.color;
                    break;
                case "Paddle":
                    if (data.x === 30) {
                        lPaddle.y = data.y;
                        lPaddle.width = data.width;
                        lPaddle.height = data.height;
                        lPaddle.color = data.color;
                        lPaddle.score = data.score;
                    }
                    else {
                        rPaddle.y = data.y;
                        rPaddle.width = data.width;
                        rPaddle.height = data.height;
                        rPaddle.color = data.color;
                        rPaddle.score = data.score;
                    }
                    break;
                default:
                    console.warn("Unknown type received:", data);
            }
        };
    })
    socket.onclose = function () { return console.log("Disconnected"); };
}
catch (error) {
    console.error("Unexpected error: ", error);
}
