import Fastify from 'fastify';
import fastifyWebsocket from '@fastify/websocket';
import { WebSocket } from "ws";
import { z } from "zod";
import path from "path";
import * as fs from 'fs';
import { fileURLToPath } from "url";
import {clearInterval} from "node:timers";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const fastify = Fastify({ logger: true });
const inputSchema = z.object({ player: z.number(), button: z.number() });
const towerSchema = z.object({ t1: z.number(), t2: z.number(), t3: z.number(), t4: z.number(), t5: z.number() });

class Game {
    level: number = 0;
    timer: Timer;
    start: boolean = false;
    state: number = 0;
    boss: boolean = false;
    constructor(timer: Timer) {
        this.timer = timer;
    }
    toJSON() {
        return {class: "Game", level: this.level, timer: this.timer.timeLeft, start: this.start, state: this.state, boss: this.boss};
    }
}

class Bullet {
    type: string;
    rank: number;
    pos: number;
    tower: Tower;
    target: Enemy;
    travel: number = 0;
    state: boolean = true;
    constructor(type: string, rank: number, pos: number, target: Enemy, tower: Tower) {
        this.type = type;
        this.rank = rank;
        this.pos = pos;
        this.target = target;
        this.tower = tower;
    }
    toJSON() {
        if (this.target && this.state)
            return {type: this.type, rank: this.rank, pos: this.pos, target: this.target.pos, travel: this.travel};
    }
}

class Timer {
    timeLeft: number;
    intervalId: ReturnType<typeof setInterval> | null = null;
    constructor(minutes: number, seconds: number) {
        this.timeLeft = minutes * 60 + seconds;
    }
    start() {
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
        }
    }
}

class Player {
    id: number;
    ws: WebSocket;
    hp: number = 3;
    mana: number = 180;
    cost: number = 50;
    enemies: Enemy[] = [];
    deck: Tower[] = [];
    board: Board[] = [];
    bullets: Bullet[] = [];
    constructor(id: number, ws: WebSocket) {
        this.id = id;
        this.ws = ws;
        for (let i = 0; i < 5; i++) {
            this.deck.push(allTowers[i].clone());
        }
    }
    addEnemy(enemy: Enemy) {
        this.enemies.push(enemy);
    }
    clearDeadEnemies() {
        this.enemies = this.enemies.filter(enemy => enemy.alive);
    }
    addBullet(bullet: Bullet) {
        this.bullets.push(bullet);
    }
    clearBullet() {
        this.bullets = this.bullets.filter(bullet => bullet.state);
    }
    spawnTower() {
        if (this.mana < this.cost || this.board.length >= 20) {
            console.log(`${this.id}: Mana insufficient or board full`);
            return ;
        }
        this.mana -= this.cost;
        this.cost += 10;
        let pos = -1;
        while (pos === -1) {
            pos = Math.floor(Math.random() * 20);
            if (this.board.find(item => item.pos === pos))
                pos = -1;
        }
        const type = Math.floor(Math.random() * this.deck.length);
        const newTower = this.deck[type].clone();
        newTower.startAttack(this, pos);
        this.board.push(new Board(pos, newTower));
        console.log(`${this.id}: Tower ${this.deck[type].type} spawned at position ${pos}`);
    }
    upTowerRank(tower: number) {
        if (this.mana >= 100 * Math.pow(2, this.deck[tower].level) && this.deck[tower].level < 4) {
            this.mana -= 100 * Math.pow(2, this.deck[tower].level);
            this.deck[tower].level += 1;
        }
    }
    toJSON() {
        return {class: "Player", id: this.id, hp: this.hp, mana: this.mana, cost: this.cost, enemies: this.enemies, deck: this.deck, board: this.board, bullets: this.bullets};
    }
}

class Tower {
    type: string;
    speed: number;
    damages: number;
    area: number;
    effect: string;
    level: number = 1;
    intervalId: ReturnType<typeof setInterval> | null = null;
    constructor(type: string, speed: number, damages: number, area: number, effect: string) {
        this.type = type;
        this.speed = speed;
        this.damages = damages;
        this.area = area;
        this.effect = effect;
    }
    attack (player: Player, enemies: Enemy[], rank: number, pos: number) {
        let maxpos = -1;
        let i: number;
        for (i = 0; i < enemies.length; i++) {
            if (enemies[i].alive && enemies[i].ihp > 0 && enemies[i].pos > maxpos) {
                maxpos = enemies[i].pos;
            }
        }
        if (maxpos === -1)
            return ;
        for (let i = 0; i < enemies.length; i++) {
            if (enemies[i].alive && enemies[i].pos >= maxpos - this.area && enemies[i].pos <= maxpos + this.area && enemies[i].ihp > 0) {
                player.addBullet(new Bullet(this.type, rank, pos, enemies[i], this));
                enemies[i].ihp -= this.damages * rank;
                if (this.area === 0)
                    break;
            }
        }
    }
    startAttack(player: Player, pos: number) {
        this.intervalId = setInterval(() => {
            const i = player.deck.findIndex(tower => tower.type === this.type);
            this.attack(player, player.enemies, player.deck[i].level, pos);
        }, 10000 / this.speed);
    }
    stopAttack() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
    clone() {
        return new Tower(this.type, this.speed, this.damages, this.area, this.effect);
    }
    toJSON() {
        return {class: "Tower", type: this.type, speed: this.speed, damages: this.damages, area: this.area, effect: this.effect, level: this.level};
    }
}

class Board {
    pos: number;
    tower: Tower;
    constructor(pos: number, tower: Tower) {
        this.tower = tower;
        this.pos = pos;
    }
    toJSON() {
        return {class: "Board", pos: this.pos, tower: this.tower};
    }
}

class Enemy {
    type: string;
    hp: number;
    ihp: number;
    speed: number;
    slow: number = 1;
    stun:boolean = false;
    pos: number = 0;
    damages: number;
    alive: boolean = true;
    constructor(type: string, hp: number, speed: number, damages: number) {
        this.type = type;
        this.hp = hp;
        this.ihp = hp;
        this.speed = speed;
        this.damages = damages;
    }
    clone () {
        return(new Enemy(this.type, this.hp, this.speed, this.damages));
    }
    toJSON() {
        if (this.alive)
            return {class: "Enemy", type: this.type, hp: this.hp, pos: this.pos};
    }
}

class RoomTd {
    id: number;
    players: Player[] = [];
    constructor (id: number, player: Player) {
        this.id = id;
        this.players.push(player);
    }
}

function loadTowers(filePath: string): Tower[] {
    const data = fs.readFileSync(filePath, 'utf-8');
    const rawTowers = JSON.parse(data);
    return (rawTowers.map((t: any) => new Tower(t.type, t.speed, t.damages, t.area, t.effect)));
}

function loadEnemies(filePath: string): Enemy[][] {
    const data = fs.readFileSync(filePath, 'utf-8');
    const jsonData = JSON.parse(data);
    return (jsonData.enemies.map((level: any[]) =>
        level.map(enemy => new Enemy(enemy.type, enemy.hp, enemy.speed, enemy.damages))
    ));
}

function generateId() {
    let newId: number;
    do {
        newId = Math.floor(Math.random() * 10000);
    } while (ids.has(newId));
    ids.add(newId);
    return newId;
}

const enemies: Enemy[][] = loadEnemies(path.join(__dirname, "../resources/enemies.json"));
const allTowers: Tower[] = loadTowers(path.join(__dirname, "../resources/towers.json"));
const ids = new Set<number>();
const roomsTd: RoomTd[] = [];

function checkGameOver(player1: Player, player2: Player, game: Game) {
    if (player1.hp <= 0 || player2.hp <= 0) {
        game.state = 2;
        player1.board.forEach((board: Board) => {
            board.tower.stopAttack()
        });
        player1.bullets.splice(0, player1.bullets.length);
        player2.board.forEach((board: Board) => {
            board.tower.stopAttack()
        });
        player2.bullets.splice(0, player2.bullets.length);
    }
}

function enemyGenerator(game: Game) {
    let wave: number;
    if (game.level != 2)
        wave = Math.floor((100 - game.timer.timeLeft) / 7);
    else
        wave = Math.floor((600 - game.timer.timeLeft) / 7);
    if (wave >= enemies[game.level].length)
        wave = enemies[game.level].length - 1;
    return (enemies[game.level][wave].clone());
}

function enemySpawner(player1: Player, player2: Player, game: Game) {
    if (game.start && game.timer.timeLeft % 8 === 7) { // Change this value to control the spawn rate (%)
        player1.addEnemy(enemyGenerator(game));
        player2.addEnemy(enemyGenerator(game));
    }
    if (game.state !== 1)
        return ;
    setTimeout(() => enemySpawner(player1, player2, game), 1000);
}

function bulletLoop(player1: Player, player2: Player) {
    player1.bullets.forEach((bullet: Bullet) => {
        bullet.travel += 1;
        if (!bullet.target || bullet.target.hp <= 0) {
            bullet.state = false;
        } else if (bullet.travel >= 100) {
            bullet.state = false;
            bullet.target.hp -= bullet.tower.damages * bullet.rank;
            if (bullet.tower.effect === "stun") {
                bullet.target.stun = true;
                setTimeout(() => {bullet.target.stun = false;}, 500);
            }
            if (bullet.tower.effect === "slow") {
                bullet.target.slow = 0.6;
                setTimeout(() => {bullet.target.slow = 1;}, 1000);
            }
        }
    })
    player1.clearBullet();
    player2.bullets.forEach((bullet: Bullet) => {
        bullet.travel += 1;
        if (!bullet.target || bullet.target.hp <= 0) {
            bullet.state = false;
        } else if (bullet.travel >= 100) {
            bullet.state = false;
            bullet.target.hp -= bullet.tower.damages * bullet.rank;
            if (bullet.tower.effect === "stun") {
                bullet.target.stun = true;
                setTimeout(() => {bullet.target.stun = false;}, 500);
            }
            if (bullet.tower.effect === "slow") {
                bullet.target.slow = 0.6;
                setTimeout(() => {bullet.target.slow = 1;}, 1000);
            }
        }
    })
    player2.clearBullet();
}

function enemyLoop(player1: Player, player2: Player, game: Game) {
    player1.enemies.forEach(enemy => {
        if (!enemy.stun)
            enemy.pos += enemy.speed * enemy.slow;
        if (enemy.hp <= 0) {
            enemy.alive = false;
            if (enemy.damages != 2) {
                player1.mana += 10;
                player2.addEnemy(enemyGenerator(game));
            } else
                player1.mana += 100;
        }
        if (enemy.pos >= 1440 && enemy.alive) {
            player1.hp -= enemy.damages;
            enemy.alive = false;
        }
    });
    player1.clearDeadEnemies();
    player2.enemies.forEach(enemy => {
        if (!enemy.stun)
            enemy.pos += enemy.speed * enemy.slow;
        if (enemy.hp <= 0) {
            enemy.alive = false;
            if (enemy.damages != 2) {
                player2.mana += 10;
                player1.addEnemy(enemyGenerator(game));
            } else
                player2.mana += 100;
        }
        if (enemy.pos >= 1440 && enemy.alive) {
            player2.hp -= enemy.damages;
            enemy.alive = false;
        }
    });
    player2.clearDeadEnemies();
    checkGameOver(player1, player2, game);
}

function gameLoop(player1: Player, player2: Player, game: Game) {
    if (game.start) {
        if (game.timer.timeLeft !== 0) {
            enemyLoop(player1, player2, game);
            bulletLoop(player1, player2);
            game.boss = false;
        } else {
            if (!game.boss) {
                const p1Board = player1.enemies.length;
                const p2Board = player2.enemies.length;
                player1.enemies.splice(0, player1.enemies.length);
                player1.bullets.splice(0, player1.bullets.length);
                player2.enemies.splice(0, player2.enemies.length);
                player2.bullets.splice(0, player2.bullets.length);
                player1.enemies.push(new Enemy("kslime", 1000 + 100 * p1Board, 1, 2)); // Boss here
                player2.enemies.push(new Enemy("kslime", 1000 + 100 * p2Board, 1, 2));
            }
            enemyLoop(player1, player2, game);
            bulletLoop(player1, player2);
            if (player1.enemies.length === 0 && player2.enemies.length === 0) {
                game.level += 1;
                if (game.level > 2)
                    game.level = 2;
                if (game.level < 2)
                    game.timer = new Timer(1, 40);
                else
                    game.timer = new Timer(6, 0);
                game.timer.start();
                player1.enemies.push(enemyGenerator(game));
                player2.enemies.push(enemyGenerator(game));
            }
            game.boss = true;
        }
    }
    if (game.state !== 1)
        return ;
    setTimeout(() => gameLoop(player1, player2, game), 10);
}

function gameInit(player1: Player, player2: Player, game: Game) {
    if (game.timer.timeLeft === 0) {
        game.timer = new Timer(1, 40); // Duree de la vague 1
        game.start = true;
        player1.addEnemy(enemies[0][0].clone());
        player2.addEnemy(enemies[0][0].clone());
        game.timer.start();
    }
    else
        setTimeout(() => gameInit(player1, player2, game), 100);
}

function mainLoop (player1: Player, player2: Player, game: Game) {
    if (game.state === 1) {
        game.timer.start();
        gameInit(player1, player2, game);
        gameLoop(player1, player2, game);
        enemySpawner(player1, player2, game);
    }
    else
        setTimeout(() => mainLoop(player1, player2, game), 100);
}

function checkRoom(room: RoomTd, intervalId: ReturnType<typeof setInterval>, game: Game) {
    if (room.players.length !== 2) {
        game.state = 2;
        clearInterval(intervalId);
        room.players.forEach(player => {
            player.ws.send(JSON.stringify({ class: "Disconnected" }))
        });
        return ;
    }
    setTimeout(() => checkRoom(room, intervalId, game), 100);
}

function leaveRoom(userId: number) {
    for (let i = 0 ; i < roomsTd.length; i++) {
        for (let j = 0; j < roomsTd[i].players.length; j++) {
            if (roomsTd[i].players[j].id === userId) {
                roomsTd[i].players.splice(j, 1);
                console.log("player id: ", userId, " left room ", roomsTd[i].id);
                return ;
            }
        }
    }
    console.log("Player has not joined a room yet.")
}

function joinRoomTd(player: Player) {
    let id: number = -1;
    let i : number = 0;
    for (; i < roomsTd.length; i++) {
        if (roomsTd[i].players.length === 1) {
            roomsTd[i].players.push(player);
            id = roomsTd[i].id;
            break ;
        }
    }
    if (id === -1) {
        let room = new RoomTd(roomsTd.length, player);
        roomsTd.push(room);
        return ;
    }
    let game = new Game(new Timer(0, 4));
    const intervalId = setInterval(() => {
        let i = roomsTd.findIndex(room => room.id === id);
        if (i === -1) {
            clearInterval(intervalId);
            return;
        }
        if (roomsTd[i].players[0]) {
            roomsTd[i].players[0].ws.send(JSON.stringify(roomsTd[i].players[0]));
            roomsTd[i].players[0].ws.send(JSON.stringify(roomsTd[i].players[1]));
            roomsTd[i].players[0].ws.send(JSON.stringify(game));
        }
        if (roomsTd[i].players[1]) {
            roomsTd[i].players[1].ws.send(JSON.stringify(roomsTd[i].players[1]));
            roomsTd[i].players[1].ws.send(JSON.stringify(roomsTd[i].players[0]));
            roomsTd[i].players[1].ws.send(JSON.stringify(game));
        }
    }, 10);
    game.state = 1;
    mainLoop(roomsTd[i].players[0], roomsTd[i].players[1], game);
    checkRoom(roomsTd[i], intervalId, game);
}

fastify.register(fastifyWebsocket);

fastify.register(async function (fastify) {
    fastify.get('/ws', {websocket: true}, (socket, req) => {
        console.log("Client connected");
        const userId = generateId();
        socket.send(JSON.stringify({class: "id", id: userId}));
        let player = new Player(userId, socket);
        allTowers.forEach((tower: Tower) => {
            socket.send(JSON.stringify(tower));
        });
        socket.on("message", (message) => {
            const msg = JSON.parse(message.toString());
            if (msg.event === "click" || msg.event === "keyDown") {
                const {data, success, error} = inputSchema.safeParse(msg);
                if (!success || !data) {
                    console.log(error);
                    return;
                }
                switch (data.button) {
                    case 5:
                        player.spawnTower();
                        break;
                    case 4:
                    case 3:
                    case 2:
                    case 1:
                    case 0:
                        player.upTowerRank(data.button);
                        break;
                    case -2:
                        player.mana += 100;
                        break;
                    default:
                        break;
                }
            } else if (msg.event === "towerInit") {
                const {data, success, error} = towerSchema.safeParse(JSON.parse(message.toString()));
                if (!success || !data) {
                    console.log(error);
                    return;
                }
                player.deck.splice(0, player.deck.length);
                player.deck.push(allTowers[data.t1].clone());
                player.deck.push(allTowers[data.t2].clone());
                player.deck.push(allTowers[data.t3].clone());
                player.deck.push(allTowers[data.t4].clone());
                player.deck.push(allTowers[data.t5].clone());
                joinRoomTd(player);
            }
        });

        socket.on("close", () => {
            leaveRoom(userId);
            console.log("Client disconnected");
        });
    });
});

fastify.listen({ port: 2246, host: '0.0.0.0' }, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
});