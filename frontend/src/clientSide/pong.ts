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
    name: string;
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
    score: string = "0";
    constructor(name: string, x: number, y: number, width: number, height: number, color: string) {
        this.name = name;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = color;
    }
}

class gameState {
    ready: boolean = false;
    state: number = 0;
    start: boolean = false;
    score1: string = "0";
    score2: string = "0";
    winner: string = "none";
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

let pongConnect: boolean;

export function Pong(mode: string, room?: number) {
    if (!room)
        room = -1;
    const canvas = document.getElementById("gameCanvas") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d")!;
    let nick = "Player 1";
    if (mode !== "local")
        nick = getNick();
    const game = new gameState();
    const ball = new Ball(canvas.width / 2, canvas.height / 2, 10, "#fcc800");
    const lPaddle = new Paddle("Player 1", 30, canvas.height / 2, 20, 200, "#fcc800");
    const rPaddle = new Paddle("Player 2", canvas.width - 30, canvas.height / 2, 20, 200, "#fcc800");
    const asset = new Assets;
    let fSize: number;
    let animFrame = 0;
    let animLoop = 1;

    function getNick(): string {
        let nick = sessionStorage.getItem('nickName');
        if (!nick) {
            nick = `guest ${Math.floor(Math.random() * 10000)}`;
            sessionStorage.setItem('nickName', nick);
        }
        return (nick);
    }

    function resizeCanvas() {
        // Récupère la taille du conteneur
        const container = document.getElementById("canvasContainer");
        if (!container) return ; // BEN ATTENTION
        const width = container.clientWidth;
        const height = width * (2 / 3); // Respecte le ratio 3:2

        // Définit la taille de la zone de dessin interne
        canvas.width = width;
        canvas.height = height;
    }

    function updatePaddle(data: Paddle) {
        const paddle = data.x === 30 ? lPaddle : rPaddle;
        paddle.name = data.name;
        paddle.x = data.x;
        paddle.y = data.y;
        paddle.width = data.width;
        paddle.height = data.height;
        paddle.color = data.color;
        paddle.score = data.score;
    }

    function updateGame(data: gameState) {
        game.state = data.state;
        game.score1 = data.score1;
        game.score2 = data.score2;
        game.start = data.start;
        game.hazard.x = data.hazard.x;
        game.hazard.y = data.hazard.y;
        game.hazard.type = data.hazard.type;
        game.timer.timeLeft = data.timer.timeLeft;
        game.timer.started = data.timer.started;
        game.winner = data.winner;
    }

    function updateBall(data: Ball) {
        ball.x = data.x;
        ball.y = data.y;
        ball.radius = data.radius;
        ball.color = data.color;
    }

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

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
        if (!game.ready) {
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
            if (mode !== "spec")
                ctx.fillText("Press any key", canvas.width * 0.5, canvas.height * 0.5 + (60 + animFrame) * ratio());
            else
                ctx.fillText("Press any to join spectator mode", canvas.width * 0.5, canvas.height * 0.5 + (60 + animFrame) * ratio());
            animFrame += animLoop;
            if (animFrame === 0 || animFrame === 9)
                animLoop *= -1;
        } else {
            fSize = Math.round(30 * ratio());
            ctx.font = `${fSize}px 'Press Start 2P'`;
            if (mode !== "spec")
                ctx.fillText("Waiting for opponent...", canvas.width * 0.5, canvas.height * 0.5 + (60 * ratio()));
            else
                ctx.fillText("Loading...", canvas.width * 0.5, canvas.height * 0.5 + (60 * ratio()));
        }

    }

    function drawBg() {
        ctx.fillStyle = "#101828";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#364153";
        for (let i = 0; i < canvas.height; i += 60) {
            ctx.fillRect(canvas.width * 0.5 - 4 * ratio(), i, 8 * ratio(), 30);
        }
    }

    function drawBall() {
        ctx.fillStyle = ball.color;
        ctx.fillRect((ball.x - ball.radius) * ratio(), (ball.y - ball.radius + 4) * ratio(), ball.radius * 2 * ratio(), (ball.radius - 4) * 2 * ratio());
        ctx.fillRect((ball.x - ball.radius + 4) * ratio(), (ball.y - ball.radius) * ratio(), (ball.radius - 4) * 2 * ratio(), ball.radius * 2 * ratio());
    }

    function drawPaddles() {
        ctx.fillStyle = lPaddle.color;
        ctx.fillRect((lPaddle.x + 3 - lPaddle.width * 0.5) * ratio(), (lPaddle.y - lPaddle.height * 0.5) * ratio(), (lPaddle.width - 6) * ratio(), lPaddle.height * ratio());
        ctx.fillRect((lPaddle.x - lPaddle.width * 0.5) * ratio(), (lPaddle.y + 3 - lPaddle.height * 0.5) * ratio(), lPaddle.width * ratio(), (lPaddle.height - 6) * ratio());
        ctx.fillStyle = rPaddle.color;
        ctx.fillRect((rPaddle.x + 3 - rPaddle.width * 0.5) * ratio(), (rPaddle.y - rPaddle.height * 0.5) * ratio(), (rPaddle.width - 6) * ratio(), rPaddle.height * ratio());
        ctx.fillRect((rPaddle.x - rPaddle.width * 0.5) * ratio(), (rPaddle.y + 3 - rPaddle.height * 0.5) * ratio(), rPaddle.width * ratio(), (rPaddle.height - 6) * ratio());
    }

    function drawScores(score1: string, score2: string) {
        ctx.fillStyle = "#fcc800";
        fSize = Math.round(48 * ratio());
        ctx.font = `${fSize}px 'Press Start 2P'`;
        ctx.textAlign = "left"
        ctx.fillText(score2, canvas.width * 0.5 + 46 * ratio(), 80 * ratio());
        ctx.textAlign = "right"
        ctx.fillText(score1, canvas.width * 0.5 - 40 * ratio(), 80 * ratio());
        // Names
        fSize = Math.round(20 * ratio());
        ctx.font = `${fSize}px 'Press Start 2P'`;
        ctx.fillText(rPaddle.name, canvas.width - (40 * ratio()), 40 * ratio());
        ctx.textAlign = "left"
        ctx.fillText(lPaddle.name, 40 * ratio(), 40 * ratio());
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
        if (game.state === 2.5) {
            ctx.fillText("You Win", canvas.width * 0.5, canvas.height * 0.4);
            fSize = Math.round(26 * ratio());
            ctx.font = `${fSize}px 'Press Start 2P'`;
            ctx.textAlign = "center";
            ctx.fillText("Opponent disconnected", canvas.width * 0.5, canvas.height * 0.60);
        } else if (mode === "local" && game.score1 > game.score2)
            ctx.fillText("Player 1 Wins", canvas.width * 0.5, canvas.height * 0.4);
        else if (mode === "local" && game.score1 < game.score2)
            ctx.fillText("Player 2 Wins", canvas.width * 0.5, canvas.height * 0.4);
        else if (mode === "remote" && game.winner === nick)
            ctx.fillText("You Win", canvas.width * 0.5, canvas.height * 0.4);
        else if (mode === "remote")
            ctx.fillText("You Lose", canvas.width * 0.5, canvas.height * 0.4);
        // ctx.fillText("Press any key to restart", canvas.width * 0.5, canvas.height * 0.70);
    }

    function specEndScreen() {
        drawBg();
        drawBall();
        drawPaddles();
        drawScores(game.score1, game.score2);
        ctx.fillStyle = "#ddae00";
        fSize = Math.round(60 * ratio());
        ctx.font = `${fSize}px 'Press Start 2P'`;
        ctx.textAlign = "center";
        ctx.fillText(`${game.winner} won`, canvas.width * 0.5, canvas.height * 0.4);
        fSize = Math.round(26 * ratio());
        ctx.font = `${fSize}px 'Press Start 2P'`;
        ctx.textAlign = "center";
        ctx.fillText("Press any key to restart", canvas.width * 0.5, canvas.height * 0.70);
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

    function oppAfkScreen() {
        ctx.fillStyle = "#364153";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#101828";
        ctx.fillRect(15 * ratio(), 15 * ratio(), canvas.width - 30 * ratio(), canvas.height - 30 * ratio());
        ctx.fillStyle = "#fcc800";
        fSize = Math.round(84 * ratio());
        ctx.font = `${fSize}px 'Press Start 2P'`;
        ctx.textAlign = "center"
        ctx.fillText("Pong Game", canvas.width * 0.5, canvas.height * 0.5);
        fSize = Math.round(30 * ratio());
        ctx.font = `${fSize}px 'Press Start 2P'`;
        ctx.fillText("Opponnent is AFK", canvas.width * 0.5, canvas.height * 0.5 + (60 * ratio()));
        ctx.fillText("you can leave now", canvas.width * 0.5, canvas.height * 0.5 + (100 * ratio()));
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
        if (mode === "spec") {
            ctx.fillStyle = "#fcc800";
            fSize = Math.round(45 * ratio());
            ctx.font = `${fSize}px 'Press Start 2P'`;
            ctx.textAlign = "center"
            ctx.fillText("Spectator Mode", canvas.width * 0.5, canvas.height * 0.95);
        }
    }

    function gameLoop() {
        switch (game.state) {
            case 0:
                titleScreen();
                break;
            case 1:
                mainLoop();
                break;
            case 2:
            case 2.5:
                endScreen();
                break;
            case 3:
                specEndScreen();
                break;
            case 4:
                oppAfkScreen();
                break;
            default:
                break;
        }
        if (pongConnect)
            requestAnimationFrame(gameLoop);
    }

    function connectionCheck(socket: WebSocket) {
        if (!pongConnect)
            socket.close();
        else
            setTimeout(() => connectionCheck(socket), 10);
    }

    try {
        const socket = new WebSocket(`match-server/ws`);
        const keyUpHandler = createKeyUpHandler(socket);
        const keyDownHandler = createKeyDownHandler(socket, game, mode);
        pongConnect = true;
        socket.onopen = function () {
            socket.send(JSON.stringify({type: "socketInit", nick: nick, room: room}));
        };

        window.addEventListener("keydown", keyDownHandler, {passive: false});
        window.addEventListener("keyup", keyUpHandler);

        socket.onmessage = function (event) {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case "gameUpdate":
                    updatePaddle(data.paddle1);
                    updatePaddle(data.paddle2);
                    updateBall(data.ball);
                    updateGame(data.game);
                    break;
                case "Disconnected":
                    if (game.state !== 2)
                        game.state = 2.5;
                    break;
                case "AFK":
                    game.state = 4;
                    break;
                case "gameEnd":
                    game.state = 3;
                    game.winner = data.winner;
                    break;
                default:
                    console.warn("Unknown type received:", data);
            }
        };
        gameLoop();
        connectionCheck(socket);
        socket.onclose = function () {
            window.removeEventListener("keyup", keyUpHandler);
            window.removeEventListener("keydown", keyDownHandler);
            window.removeEventListener("resize", resizeCanvas);
            return
        };
    } catch (error) {
        console.error("Unexpected error: ", error);
    }
}

function createKeyUpHandler(socket: WebSocket) {
    return function(event: KeyboardEvent) {
        socket.send(JSON.stringify({ type: "input", key: event.key, state: "up" }));
    };
}

function createKeyDownHandler(socket: WebSocket, game: gameState, mode: string) {
    return function(event: KeyboardEvent) {
        if (["ArrowUp", "ArrowDown"].includes(event.key)) {
            event.preventDefault();
        }
        if (game.state !== 2 && game.state !== 2.5) {
            if (!game.ready) {
                game.ready = true;
                socket.send(JSON.stringify({type: "ready", mode: mode}));
            } else {
                socket.send(JSON.stringify({type: "input", key: event.key, state: "down"}));
                if (game.state === 3)
                    game.state = 0;
            }
        } else {
            pongConnect = false;
        }
    };
}

export function pongStop() {
    pongConnect = false;
}