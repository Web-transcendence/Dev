const canvas = document.getElementById("gameCanvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
let connect: boolean = true;

function resizeCanvas() {
    // Récupère la taille du conteneur
    const container = document.getElementById("canvasContainer");
    const width = container.clientWidth;
    const height = width * (2 / 3); // Respecte le ratio 3:2

    // Définit la taille de la zone de dessin interne
    canvas.width = width;
    canvas.height = height;
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();

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
    timer: Timer = new Timer(0, 4);
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

class Timer {
    timeLeft: number;
    started: boolean = false
    constructor(minutes: number, seconds: number) {
        this.timeLeft = minutes * 60 + seconds;
    }

}

class Assets {
    BarUp: HTMLImageElement = new Image();
    BarDown: HTMLImageElement = new Image();
    BallUp: HTMLImageElement = new Image();
    constructor () {
        this.BarUp.src = "./assets/pong/barup.png";
        this.BarDown.src = "./assets/pong/bardown.png";
        this.BallUp.src = "./assets/pong/ballup.png";
    }
}

let animFrame = 0;
let animLoop = 1;
let game = new gameState();
let ball = new Ball(canvas.width / 2, canvas.height / 2, 10, "#fcc800");
let lPaddle = new Paddle(30, canvas.height / 2, 20, 200, "#fcc800");
let rPaddle = new Paddle(canvas.width - 30, canvas.height / 2, 20, 200, "#fcc800");
let asset = new Assets;
let fSize: number;
let ready = false;

function ratio() {
    return (canvas.width / 1200);
}

function titleScreen() {
    ctx.fillStyle = "#364153";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#101828";
    ctx.fillRect(15 * ratio(), 15 * ratio(), canvas.width - 30 * ratio(), canvas.height - 30 * ratio());
    ctx.fillStyle = "#fcc800";
    fSize = Math.round(84 * ratio());
    ctx.font = `${fSize}px 'Press Start 2P'`;
    ctx.textAlign = "center"
    ctx.fillText("Pong Game", canvas.width * 0.5, canvas.height * 0.5);
    if (!ready) {
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
        fSize = Math.round(30 * ratio());
        ctx.font = `${fSize}px 'Press Start 2P'`;
        ctx.fillText("Press any key", canvas.width * 0.5, canvas.height * 0.5 + (60 + animFrame) * ratio());
        animFrame += animLoop;
        if (animFrame === 0 || animFrame === 9)
            animLoop *= -1;
    }
    else {
        fSize = Math.round(30 * ratio());
        ctx.font = `${fSize}px 'Press Start 2P'`;
        ctx.fillText("Waiting for opponent...", canvas.width * 0.5, canvas.height * 0.5 + (60 * ratio()));
    }

}

function drawBg () {
    ctx.fillStyle = "#101828";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#364153";
    for (let i = 0; i < canvas.height; i += 60) {
        ctx.fillRect(canvas.width * 0.5 - 4 * ratio(), i, 8 * ratio(), 30);
    }
}

function drawBall () {
    ctx.fillStyle = ball.color;
    ctx.fillRect((ball.x - ball.radius) * ratio(), (ball.y - ball.radius + 4) * ratio(), ball.radius * 2 * ratio(), (ball.radius - 4) * 2 * ratio());
    ctx.fillRect((ball.x - ball.radius + 4) * ratio(), (ball.y - ball.radius) * ratio(), (ball.radius - 4) * 2 * ratio(), ball.radius * 2 * ratio());
}

function drawPaddles () {
    ctx.fillStyle = lPaddle.color;
    ctx.fillRect((lPaddle.x + 3 - lPaddle.width * 0.5) * ratio(), (lPaddle.y - lPaddle.height * 0.5) * ratio(), (lPaddle.width - 6) * ratio(), lPaddle.height * ratio());
    ctx.fillRect((lPaddle.x - lPaddle.width * 0.5) * ratio(), (lPaddle.y + 3 - lPaddle.height * 0.5) * ratio(), lPaddle.width * ratio(), (lPaddle.height - 6) * ratio());
    ctx.fillStyle = rPaddle.color;
    ctx.fillRect((rPaddle.x + 3 - rPaddle.width * 0.5) * ratio(), (rPaddle.y - rPaddle.height * 0.5) * ratio(), (rPaddle.width - 6) * ratio(), rPaddle.height * ratio());
    ctx.fillRect((rPaddle.x - rPaddle.width * 0.5) * ratio(), (rPaddle.y + 3 - rPaddle.height * 0.5) * ratio(), rPaddle.width * ratio(), (rPaddle.height - 6) * ratio());
}

function drawScores (score1: string, score2: string) {
    ctx.fillStyle = "#fcc800";
    fSize = Math.round(48 * ratio());
    ctx.font = `${fSize}px 'Press Start 2P'`;
    ctx.textAlign = "left"
    ctx.fillText(score2, canvas.width * 0.5 + 46 * ratio(), 80 * ratio());
    ctx.textAlign = "right"
    ctx.fillText(score1, canvas.width * 0.5 - 40 * ratio(), 80 * ratio());
}

function endScreen() {
    drawBg();
    drawBall();
    drawPaddles();
    drawScores(game.score1, game.score2);
    ctx.fillStyle = "#ddae00";
    fSize = Math.round(60 * ratio());
    ctx.font = `${fSize}px 'Press Start 2P'`;
    ctx.textAlign = "center";
    if (game.score1 > game.score2)
        ctx.fillText("Player 1 Wins", canvas.width * 0.5, canvas.height * 0.4);
    else
        ctx.fillText("Player 2 Wins", canvas.width * 0.5, canvas.height * 0.4);
    fSize = Math.round(26 * ratio());
    ctx.font = `${fSize}px 'Press Start 2P'`;
    ctx.textAlign = "center";
    ctx.fillText("Press any key to restart game", canvas.width * 0.5, canvas.height * 0.65);
}

function drawHazard() {
    switch (game.hazard.type) {
        case "BarSizeUp":
            ctx.drawImage(asset.BarUp, (game.hazard.x - 25) * ratio(), (game.hazard.y - 25) * ratio(), 50 * ratio(), 50 * ratio());
            break;
        case "BarSizeDown":
            ctx.drawImage(asset.BarDown, (game.hazard.x - 25) * ratio(), (game.hazard.y - 25) * ratio(), 50 * ratio(), 50 * ratio());
            break;
        case "BallSpeedUp":
            ctx.drawImage(asset.BallUp, (game.hazard.x - 25) * ratio(), (game.hazard.y - 25) * ratio(), 50 * ratio(), 50 * ratio());
            break;
        default:
            break;
    }
}

function mainLoop() {
    drawBg();
    drawBall();
    drawScores(lPaddle.score, rPaddle.score);
    drawHazard();
    drawPaddles();
    // Timer
    if (game.timer.started) {
        ctx.fillStyle = "#fcc800";
        fSize = Math.round(60 * ratio());
        ctx.font = `${fSize}px 'Press Start 2P'`;
        ctx.textAlign = "center"
        ctx.fillText(game.timer.timeLeft.toString(), canvas.width * 0.5, canvas.height * 0.75);
    }
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
            if (!connect)
                return;
            ready = true;
            socket.send(JSON.stringify({ type: "input", key: event.key, state: "down" }));
        });
        window.addEventListener("keyup", (event) => {
            if (!connect)
                return ;
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
                    game.timer.timeLeft = data.timer.timeLeft;
                    game.timer.started = data.timer.started;
                    break;
                case "Ball":
                    ball.x = data.x;
                    ball.y = data.y;
                    ball.radius = data.radius;
                    ball.color = data.color;
                    break;
                case "Paddle":
                    if (data.x === 30) {
                        lPaddle.x = data.x;
                        lPaddle.y = data.y;
                        lPaddle.width = data.width;
                        lPaddle.height = data.height;
                        lPaddle.color = data.color;
                        lPaddle.score = data.score;
                    }
                    else {
                        rPaddle.x = data.x;
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
