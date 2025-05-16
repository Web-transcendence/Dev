// Classes
export class Bullet {
    type: string;
    rank: number;
    pos: number;
    target: number;
    travel: number;

    constructor(type: string, rank: number, pos: number, target: number, travel: number) {
        this.type = type;
        this.rank = rank;
        this.pos = pos;
        this.target = target;
        this.travel = travel;
    }
}

export class GameTd {
    level: number = 0;
    timer: number = 4;
    start: boolean = false;
    state: number = 0;
    boss: boolean = false;
}

export class Enemy {
    type: string;
    hp: number;
    pos: number;

    constructor(type: string, hp: number, pos: number) {
        this.type = type;
        this.hp = hp;
        this.pos = pos;
    }
}

export class Player {
    name: string;
    id: number = -1;
    hp: number = 3;
    mana: number = 210;
    cost: number = 60;
    enemies: Enemy[] = [];
    deck: Tower[] = [];
    board: Board[] = [];
    bullets: Bullet[] = [];

    constructor(name: string) {
        this.name = name;
    }

    addEnemy(enemy: Enemy) {
        this.enemies.push(enemy);
    }

    addBullet(bullet: Bullet) {
        this.bullets.push(bullet);
    }
}

export class Tower {
    type: string;
    speed: number;
    damages: number;
    area: number;
    effect: string;
    level: number;
    pos: number = 0;

    constructor(type: string, speed: number, damages: number, area: number, effect: string, level: number) {
        this.type = type;
        this.speed = speed;
        this.damages = damages;
        this.area = area;
        this.effect = effect;
        this.level = level;
    }
}

export class Board {
    pos: number;
    tower: Tower;

    constructor(pos: number, tower: Tower) {
        this.tower = tower;
        this.pos = pos;
    }
}

export class AssetsTd {
    images: Record<string, HTMLImageElement> = {};
    frame: number = 0;
    async load(): Promise<void> {
        const assetsFolder = "./assets/tower-defense/";
        const imageNames = ["black1.png", "black2.png", "black3.png", "black4.png",
            "blue1.png", "blue2.png", "blue3.png", "blue4.png",
            "green1.png", "green2.png", "green3.png", "green4.png",
            "orange1.png", "orange2.png", "orange3.png", "orange4.png",
            "pink1.png", "pink2.png", "pink3.png", "pink4.png",
            "red1.png", "red2.png", "red3.png", "red4.png",
            "violet1.png", "violet2.png", "violet3.png", "violet4.png",
            "white1.png", "white2.png", "white3.png", "white4.png",
            "yellow1.png", "yellow2.png", "yellow3.png", "yellow4.png",
            "ygreen1.png", "ygreen2.png", "ygreen3.png", "ygreen4.png",
            "main.png", "addTower.png", "hp.png", "mana.png", "random.png", "randomh.png",
            "map0.png", "map1.png", "map2.png", "map3.png", "map4.png",
            "poop0.png", "poop1.png",
            "bslime0.png", "bslime1.png",
            "gslime0.png", "gslime1.png",
            "yslime0.png", "yslime1.png",
            "rslime0.png", "rslime1.png",
            "pslime0.png", "pslime1.png",
            "dslime0.png", "dslime1.png",
            "kslime0.png", "kslime1.png"];

        const promises = imageNames.map(name => {
            return new Promise<void>((resolve, reject) => {
                const key = name.split(".")[0];
                const img = new Image();
                img.src = `${assetsFolder}${name}`;
                img.onload = () => {
                    this.images[key] = img;
                    resolve();
                };
                img.onerror = (err) => {
                    console.error(`Erreur de chargement pour l'image ${name}`, err);
                    reject(err);
                };
            });
        });

        await Promise.all(promises);
    }

    getImage(name: string): HTMLImageElement | undefined {
        return this.images[name];
    }

    getAnImage(name: string): HTMLImageElement | undefined {
        if (this.frame % 30 < 15 && this.images[`${name}0`]) {
            return this.images[`${name}0`];
        } else if (this.images[`${name}1`]) {
            return this.images[`${name}1`];
        }
        return this.images[name];
    }
}

export let tdConnect: boolean;

// Main function
export function TowerDefense(room?: number) {
    // Global Variables
    if (!room)
        room = -1;
    const canvasTd = document.getElementById("gameCanvas") as HTMLCanvasElement;
    const ctxTd = canvasTd.getContext("2d")!;
    const assetsTd = new AssetsTd();
    const tile = canvasTd.width / 15;
    const gameTd = new GameTd;
    const nick = getNick();
    const player1 = new Player(nick);
    const player2 = new Player("Player 2");
    const nmap = Math.floor(Math.random() * 5);
    const allTowers: Tower[] = [];
    const selected: number[] = [];
    let rdmhover = false;

    function getNick(): string {
        let nick = sessionStorage.getItem('nickName');
        if (!nick) {
            nick = `guest${Math.floor(Math.random() * 10000)}`;
            sessionStorage.setItem('nickName', nick);
        }
        return (nick);
    }

    // Main menu and tower selection
    function aoeornot(area: number) {
        if (area !== 0)
            return ("area dmg");
        return ("mono dmg");
    }

    function isSelected(i: number) {
        if (selected.includes(i))
            return ("#b329d1");
        return ("#eaeaea");
    }

    function dots() {
        if (assetsTd.frame % 120 < 30)
            return (".");
        if (assetsTd.frame % 120 < 60)
            return ("..");
        if (assetsTd.frame % 120 < 90)
            return ("...");
        return ("");
    }

    function drawMenu() {
        // Background
        ctxTd.drawImage(assetsTd.getImage("main")!, 0, 0, canvasTd.width, canvasTd.height);
        // Title
        ctxTd.drawImage(assetsTd.getAnImage("rslime")!, canvasTd.width * 0.1, canvasTd.height * 0.06, 170, 170);
        ctxTd.drawImage(assetsTd.getAnImage("gslime")!, canvasTd.width * 0.18, canvasTd.height * 0.17, 80, 80);
        ctxTd.drawImage(assetsTd.getAnImage("pslime")!, canvasTd.width * 0.11, canvasTd.height * 0.18, 80, 80);
        assetsTd.frame += 0.75;
        ctxTd.fillStyle = "#b329d1";
        ctxTd.strokeStyle = "#0d0d0d";
        ctxTd.lineWidth = tile / 4;
        ctxTd.textAlign = "center";
        ctxTd.font = "64px 'Press Start 2P'";
        ctxTd.strokeText("Slime Defender", canvasTd.width * 0.545, canvasTd.height * 0.25, canvasTd.width * 0.6);
        ctxTd.fillText("Slime Defender", canvasTd.width * 0.545, canvasTd.height * 0.25, canvasTd.width * 0.6);
        // Tower Selection
        for (let i = 0; i < allTowers.length; i++) {
            let centerx = (0.18 + Math.floor(i % 5) * 0.16) * canvasTd.width;
            let centery = (0.54 + Math.floor(i / 5) * 0.16) * canvasTd.height;
            drawRawButton(centerx, centery, canvasTd.width * 0.15, canvasTd.height * 0.15, isSelected(i));
            ctxTd.drawImage(assetsTd.getImage(`${allTowers[i].type}4`)!, centerx - tile * 1.1, centery - tile * 0.6, canvasTd.height * 0.11, canvasTd.height * 0.11);
            ctxTd.fillStyle = "#eaeaea";
            ctxTd.textAlign = "left";
            ctxTd.font = `${tile / 5}px 'Press Start 2P'`;
            ctxTd.fillText(`atk: ${allTowers[i].damages}`, centerx, centery - tile * 0.35, canvasTd.width * 0.06);
            ctxTd.fillText(`spd: ${allTowers[i].speed}`, centerx, centery - tile * 0.05, canvasTd.width * 0.06);
            ctxTd.fillText(aoeornot(allTowers[i].area), centerx, centery + tile * 0.25, canvasTd.width * 0.06);
            if (allTowers[i].effect !== "none")
                ctxTd.fillText(allTowers[i].effect, centerx, centery + tile * 0.55, canvasTd.width * 0.06);
        }
        // Random button
        drawRawButton(canvasTd.width * 0.63, canvasTd.height * 0.9, canvasTd.width * 0.06, canvasTd.height * 0.06, "#eaeaea");
        if (rdmhover)
            ctxTd.drawImage(assetsTd.getImage("randomh")!, canvasTd.width * 0.634, canvasTd.height * 0.9 - tile * 0.125, tile * 0.25, tile * 0.25);
        else
            ctxTd.drawImage(assetsTd.getImage("random")!, canvasTd.width * 0.634, canvasTd.height * 0.9 - tile * 0.125, tile * 0.25, tile * 0.25);
        // Start button
        ctxTd.textAlign = "center";
        ctxTd.font = `${tile / 4.2}px 'Press Start 2P'`;
        if (gameTd.state === 0.5) {
            drawRawButton(canvasTd.width * 0.5, canvasTd.height * 0.9, canvasTd.width * 0.26, canvasTd.height * 0.1, "#b329d1");
            ctxTd.fillText("Waiting for opponent", canvasTd.width * 0.5, canvasTd.height * 0.91, canvasTd.width * 0.22);
            ctxTd.fillText(dots(), canvasTd.width * 0.5, canvasTd.height * 0.935);
        } else if (selected.length === 5) {
            drawRawButton(canvasTd.width * 0.5, canvasTd.height * 0.9, canvasTd.width * 0.26, canvasTd.height * 0.1, "#b329d1");
            ctxTd.fillText("Click to start", canvasTd.width * 0.5, canvasTd.height * 0.91, canvasTd.width * 0.22);
        } else {
            drawRawButton(canvasTd.width * 0.5, canvasTd.height * 0.9, canvasTd.width * 0.26, canvasTd.height * 0.1, "#eaeaea");
            ctxTd.fillText("Select 5 rocks", canvasTd.width * 0.5, canvasTd.height * 0.91, canvasTd.width * 0.22);
        }
    }

    // Game
    function timeTostring(timer: number) {
        const minutes = Math.floor(timer / 60);
        const seconds = timer % 60;
        return (`${minutes}:${seconds.toString().padStart(2, '0')}`);
    }

    function drawBullets() {
        let tPosx;
        let tPosy;
        let ePosx;
        let ePosy;
        player1.bullets.forEach(bullet => {
            tPosx = tile * (1.45 + bullet.pos % 4);
            tPosy = tile * (2.2 + Math.floor(bullet.pos / 4));
            ePosx = enemyPosx(bullet.target, 1);
            ePosy = enemyPosy(bullet.target);
            ctxTd.fillStyle = getTowerColor(bullet.type);
            ctxTd.beginPath();
            ctxTd.arc(tPosx + (ePosx - tPosx) * bullet.travel / 100, tPosy + (ePosy - tPosy) * bullet.travel / 100, bullet.rank * 4, 0, 2 * Math.PI);
            ctxTd.fill();
        });
        player2.bullets.forEach(bullet => {
            tPosx = tile * (10.45 + bullet.pos % 4);
            tPosy = tile * (2.2 + Math.floor(bullet.pos / 4));
            ePosx = enemyPosx(bullet.target, 2);
            ePosy = enemyPosy(bullet.target);
            ctxTd.fillStyle = getTowerColor(bullet.type);
            ctxTd.beginPath();
            ctxTd.arc(tPosx + (ePosx - tPosx) * bullet.travel / 100, tPosy + (ePosy - tPosy) * bullet.travel / 100, bullet.rank * 4, 0, 2 * Math.PI);
            ctxTd.fill();
        });
    }

    function drawTimer() {
        switch (nmap) {
            case 4:
                ctxTd.fillStyle = "#dfdfdf";
                break;
            case 3:
                ctxTd.fillStyle = "#a08829";
                break;
            case 2:
                ctxTd.fillStyle = "#ab1e00";
                break;
            case 1:
                ctxTd.fillStyle = "#17b645";
                break;
            case 0:
                ctxTd.fillStyle = "#0075ab";
                break;
            default:
                ctxTd.fillStyle = "#fcc800";
                break;
        }
        ctxTd.strokeStyle = "#0d0d0d";
        ctxTd.lineWidth = tile * 0.1;
        ctxTd.font = `${tile * 0.5}px 'Press Start 2P'`;
        ctxTd.textAlign = "center"
        let timer = gameTd.timer;
        if (gameTd.start) {
            if (!gameTd.boss) {
                ctxTd.strokeText(timeTostring(timer), canvasTd.width * 0.5, canvasTd.height * 0.5 + tile * 0.25);
                ctxTd.fillText(timeTostring(timer), canvasTd.width * 0.5, canvasTd.height * 0.5 + tile * 0.25);
            } else {
                ctxTd.strokeText("Boss", canvasTd.width * 0.5, canvasTd.height * 0.5 + tile * 0.25);
                ctxTd.fillText("Boss", canvasTd.width * 0.5, canvasTd.height * 0.5 + tile * 0.25);
            }
        } else if (timer > 1) {
            ctxTd.strokeText(timeTostring(timer - 1), canvasTd.width * 0.5, canvasTd.height * 0.5 + tile * 0.25);
            ctxTd.fillText(timeTostring(timer - 1), canvasTd.width * 0.5, canvasTd.height * 0.5 + tile * 0.25);
        } else {
            ctxTd.strokeText("Go !!!", canvasTd.width * 0.5, canvasTd.height * 0.5 + 23);
            ctxTd.fillText("Go !!!", canvasTd.width * 0.5, canvasTd.height * 0.5 + 23);
        }
    }

    function enemyPosx(pos: number, player: number) {
        if (pos < 480) {
            if (player === 1)
                return (pos);
            else
                return (canvasTd.width - pos);
        }
        if (pos > 1040) {
            if (player === 1)
                return (tile * 19 - pos);
            else
                return (pos - tile * 4);
        } else {
            if (player === 1)
                return (tile * 6 - 1);
            else
                return (tile * 9 - 1);
        }
    }

    function enemyPosy(pos: number) {
        if (pos < 480) {
            return (tile - 1);
        }
        if (pos > 1040) {
            return (tile * 8 - 1);
        } else
            return (pos - tile * 5 - 1);
    }

    function drawEnemies() {
        player1.enemies.forEach(enemy => {
            if (enemy.pos < 480)
                ctxTd.drawImage(assetsTd.getAnImage(enemy.type)!, enemyPosx(enemy.pos, 1) - tile * 0.5, enemyPosy(enemy.pos) - tile * 0.5, tile, tile);
            else {
                ctxTd.save();
                ctxTd.scale(-1, 1);
                ctxTd.drawImage(assetsTd.getAnImage(enemy.type)!, -1 * (enemyPosx(enemy.pos, 1) - tile * 0.5) - 70, enemyPosy(enemy.pos) - tile * 0.5, tile, tile);
                ctxTd.restore();
            }
            ctxTd.fillStyle = "#eaeaea";
            ctxTd.font = "16px 'Press Start 2P'";
            ctxTd.textAlign = "center";
            ctxTd.fillText(enemy.hp.toString(), enemyPosx(enemy.pos, 1), enemyPosy(enemy.pos) + 28);
        });
        player2.enemies.forEach(enemy => {
            if (enemy.pos >= 480)
                ctxTd.drawImage(assetsTd.getAnImage(enemy.type)!, enemyPosx(enemy.pos, 2) - tile * 0.5, enemyPosy(enemy.pos) - tile * 0.5, tile, tile);
            else {
                ctxTd.save();
                ctxTd.scale(-1, 1);
                ctxTd.drawImage(assetsTd.getAnImage(enemy.type)!, -1 * (enemyPosx(enemy.pos, 2) - tile * 0.5) - 70, enemyPosy(enemy.pos) - tile * 0.5, tile, tile);
                ctxTd.restore();
            }
            ctxTd.fillStyle = "#eaeaea";
            ctxTd.font = "16px 'Press Start 2P'";
            ctxTd.textAlign = "center";
            ctxTd.fillText(enemy.hp.toString(), enemyPosx(enemy.pos, 2), enemyPosy(enemy.pos) + 28);
        });
        assetsTd.frame += 1;
    }

    function getTowerColor(type: string) {
        if (type === "red")
            return ("#ab1e00");
        if (type === "blue")
            return ("#0075ab");
        if (type === "green")
            return ("#17b645");
        if (type === "yellow")
            return ("#ffe71e");
        if (type === "white")
            return ("#dfdfdf");
        if (type === "black")
            return ("#030303")
        if (type === "orange")
            return ("#ff8000")
        if (type === "pink")
            return ("#e74fff")
        if (type === "violet")
            return ("#b329d1")
        if (type === "ygreen")
            return ("#b8dc04")
        return ("black");
    }

    function drawRawButton(centerx: number, centery: number, sizex: number, sizey: number, border: string) {
        const old = ctxTd.fillStyle;
        ctxTd.fillStyle = border;
        ctxTd.fillRect(centerx - sizex * 0.5 + 4, centery - sizey * 0.5, sizex - 8, sizey);
        ctxTd.fillRect(centerx - sizex * 0.5, centery - sizey * 0.5 + 4, sizex, sizey - 8);
        ctxTd.fillStyle = "#0d0d0d";
        ctxTd.fillRect(centerx - sizex * 0.5 + 8, centery - sizey * 0.5 + 4, sizex - 16, sizey - 8);
        ctxTd.fillRect(centerx - sizex * 0.5 + 4, centery - sizey * 0.5 + 8, sizex - 8, sizey - 16);
        ctxTd.fillStyle = old;
    }

    function drawButtons() {
        ctxTd.fillStyle = "#fcc800";
        ctxTd.font = "16px 'Press Start 2P'";
        ctxTd.textAlign = "center";
        // addTower
        ctxTd.drawImage(assetsTd.getImage("addTower")!, tile * 6.5 - 35, canvasTd.height - tile * 0.75 - 35, 70, 70);
        ctxTd.fillText(player1.cost.toString(), tile * 6.5, canvasTd.height - tile * 0.75 + 22);
        // stats
        drawRawButton(tile * 5.5, canvasTd.height - tile * 0.75, tile * 0.9, tile * 0.9, "#0096ff");
        ctxTd.drawImage(assetsTd.getImage("mana")!, tile * 5.35, canvasTd.height - tile * 1.05, tile * 0.3, tile * 0.3);
        drawRawButton(tile * 9.5, canvasTd.height - tile * 0.75, tile * 0.9, tile * 0.9, "#0096ff");
        ctxTd.drawImage(assetsTd.getImage("mana")!, tile * 9.35, canvasTd.height - tile * 1.05, tile * 0.3, tile * 0.3);
        ctxTd.fillStyle = "#0096ff";
        ctxTd.fillText(player1.mana.toString(), tile * 5.5, canvasTd.height - tile * 0.45, 50);
        ctxTd.fillText(player2.mana.toString(), tile * 9.5, canvasTd.height - tile * 0.45, 50);
        // Towers
        for (let i = 0; i < player1.deck.length && i < player2.deck.length; i++) {
            // Player 1
            drawRawButton(tile * (0.5 + i), canvasTd.height - tile * 0.75, tile * 0.9, tile * 1.3, getTowerColor(player1.deck[i].type));
            ctxTd.drawImage(assetsTd.getImage(`${player1.deck[i].type}${player1.deck[i].level}`)!, tile * (0.5 + i) - 28, canvasTd.height - tile - 28, 56, 56);
            ctxTd.fillStyle = "#eaeaea";
            if (player1.deck[i].level !== 4) {
                ctxTd.fillText(`Lv. ${player1.deck[i].level}`, tile * (0.5 + i), canvasTd.height - tile * 0.55, 50);
                ctxTd.fillText(`Up: ${100 * Math.pow(2, player1.deck[i].level)}`, tile * (0.5 + i), canvasTd.height - tile * 0.25, 50);
            } else
                ctxTd.fillText(`Lv. max`, tile * (0.5 + i), canvasTd.height - tile * 0.35, 50);
            // Player 2
            drawRawButton(tile * (10.5 + i), canvasTd.height - tile * 0.75, tile * 0.9, tile * 1.3, getTowerColor(player2.deck[i].type));
            ctxTd.fillStyle = "#eaeaea";
            ctxTd.drawImage(assetsTd.getImage(`${player2.deck[i].type}${player2.deck[i].level}`)!, tile * (10.5 + i) - 28, canvasTd.height - tile - 28, 56, 56);
            if (player2.deck[i].level !== 4) {
                ctxTd.fillText(`Lv. ${player2.deck[i].level}`, tile * (10.5 + i), canvasTd.height - tile * 0.55, 50);
                ctxTd.fillText(`Up: ${100 * Math.pow(2, player2.deck[i].level)}`, tile * (10.5 + i), canvasTd.height - tile * 0.25, 50);
            } else
                ctxTd.fillText(`Lv. max`, tile * (10.5 + i), canvasTd.height - tile * 0.35, 50);
        }
        // HP
        for (let j = 0; j < player1.hp; j++) {
            ctxTd.drawImage(assetsTd.getImage("hp")!, tile * (0.1 + j * 0.6), tile * 0.1, tile * 0.5, tile * 0.5);
        }
        for (let k = 0; k < player2.hp; k++) {
            ctxTd.drawImage(assetsTd.getImage("hp")!, canvasTd.width - tile * (0.6 + k * 0.6), tile * 0.1, tile * 0.5, tile * 0.5);
        }
        // Names
        ctxTd.strokeStyle = "#0d0d0d";
        ctxTd.lineWidth = tile * 0.08;
        ctxTd.textAlign = "left";
        ctxTd.strokeText(player1.name, tile * 1.9, tile * 0.45, tile * 3);
        ctxTd.fillText(player1.name, tile * 1.9, tile * 0.45, tile * 3);
        ctxTd.textAlign = "right";
        ctxTd.strokeText(player2.name, canvasTd.width - tile * 1.9, tile * 0.45, tile * 3);
        ctxTd.fillText(player2.name, canvasTd.width - tile * 1.9, tile * 0.45, tile * 3);
    }

    function getTowerLevel(tower: Tower, pNum: number) {
        const player: Player = pNum === 1 ? player1 : player2;
        for (let i = 0; i < player.deck.length; i++) {
            if (player.deck[i].type === tower.type) {
                return (player.deck[i].level);
            }
        }
    }

    function drawTowers() {
        player1.board.forEach(tower => {
            ctxTd.drawImage(assetsTd.getImage(`${tower.tower.type}${getTowerLevel(tower.tower, 1)}`)!, tile * (0.75 + tower.pos % 4), tile * (1.5 + Math.floor(tower.pos / 4)), tile * 1.4, tile * 1.4);
        });
        player2.board.forEach(tower => {
            ctxTd.drawImage(assetsTd.getImage(`${tower.tower.type}${getTowerLevel(tower.tower, 2)}`)!, tile * (9.75 + tower.pos % 4), tile * (1.5 + Math.floor(tower.pos / 4)), tile * 1.4, tile * 1.4);
        });
    }

    function drawGrid() {
        let sq = true;
        for (let i = 0; i < canvasTd.width; i += tile) {
            let j = 0;
            sq = !sq;
            for (; j < canvasTd.height; j += tile) {
                if (sq) {
                    ctxTd.fillStyle = "#364153";
                    sq = false;
                } else {
                    ctxTd.fillStyle = "#101828";
                    sq = true;
                }
                ctxTd.fillRect(i, j, tile, tile);
            }
        }
    }

    function drawTemplate() {
        ctxTd.fillStyle = "red";
        ctxTd.fillRect(0, tile * 0.5, tile * 6.5, tile * 8);
        ctxTd.fillStyle = "blue";
        ctxTd.strokeRect(0, tile, 6 * tile, 7 * tile);
        ctxTd.fillStyle = "brown";
        ctxTd.fillRect(tile * 0.5, tile * 1.5, tile * 5, tile * 6);
        ctxTd.fillStyle = "red";
        ctxTd.fillRect(0, tile * 7.5, tile, tile);
        ctxTd.fillStyle = "green";
        ctxTd.fillRect(tile, tile * 2, tile * 4, tile * 5);
    }

    function drawGameTd() {
        ctxTd.drawImage(assetsTd.getImage(`map${nmap}`)!, 0, 0, canvasTd.width, canvasTd.height);
        //drawGrid(); // for debug use only
        //drawTemplate(); // for debug use only
        drawTimer();
        drawEnemies();
        drawButtons();
        drawTowers();
        drawBullets();
    }

    // EndScreen
    function drawEndScreen() {
        ctxTd.drawImage(assetsTd.getImage(`map${nmap}`)!, 0, 0, canvasTd.width, canvasTd.height);
        drawEnemies();
        drawButtons();
        drawTowers();
        ctxTd.strokeStyle = "#0d0d0d";
        ctxTd.lineWidth = tile * 0.2;
        ctxTd.font = `${tile}px 'Press Start 2P'`;
        ctxTd.textAlign = "center"
        if (gameTd.state === 2.5) {
            ctxTd.fillStyle = "#17b645";
            ctxTd.strokeText("You win!", canvasTd.width * 0.5, canvasTd.height * 0.5);
            ctxTd.fillText("You win!", canvasTd.width * 0.5, canvasTd.height * 0.5);
            ctxTd.font = `${tile / 2}px 'Press Start 2P'`;
            ctxTd.lineWidth = tile * 0.1;
            ctxTd.strokeText("Opponent disconnected", canvasTd.width * 0.5, canvasTd.height * 0.6);
            ctxTd.fillText("Opponent disconnected", canvasTd.width * 0.5, canvasTd.height * 0.6);
        } else if (player1.hp > player2.hp) {
            ctxTd.fillStyle = "#17b645";
            ctxTd.strokeText("You win!", canvasTd.width * 0.5, canvasTd.height * 0.5);
            ctxTd.fillText("You win!", canvasTd.width * 0.5, canvasTd.height * 0.5);
        } else if (player2.hp > player1.hp) {
            ctxTd.fillStyle = "#ab1e00";
            ctxTd.strokeText("You lose!", canvasTd.width * 0.5, canvasTd.height * 0.5);
            ctxTd.fillText("You lose!", canvasTd.width * 0.5, canvasTd.height * 0.5);
        } else {
            ctxTd.fillStyle = "#0075ab";
            ctxTd.strokeText("Draw!", canvasTd.width * 0.5, canvasTd.height * 0.5);
            ctxTd.fillText("Draw!", canvasTd.width * 0.5, canvasTd.height * 0.5);
        }
    }

    // Loop
    function mainLoopTd() {
        switch (gameTd.state) {
            case 0:
            case 0.5:
                drawMenu();
                break;
            case 1:
                drawGameTd();
                break;
            case 2:
            case 2.5:
                drawEndScreen();
                break;
            default:
                break;
        }
        if (tdConnect)
            requestAnimationFrame(mainLoopTd);
    }

    function connectionCheck(socket: WebSocket) {
        if (!tdConnect)
            socket.close();
        else
            setTimeout(() => connectionCheck(socket), 10);
    }

    // Main
    assetsTd.load().then(() => {
        console.log("Toutes les images sont chargées!");
        mainLoopTd();
    }).catch(error => {
        console.error("Erreur lors du chargement des assets: ", error);
    });

    // Communication with backend
    function updatePlayer(data: Player) {
        let board: Board;
        let i: number;
        const player = data.id === player1.id ? player1 : player2;
        player.name = data.name;
        player.hp = data.hp;
        player.mana = data.mana;
        player.cost = data.cost;
        player.enemies.splice(0, player.enemies.length);
        data.enemies.forEach((enemy: Enemy) => {
            if (enemy)
                player.enemies.push(new Enemy(enemy.type, enemy.hp, enemy.pos));
        });
        player.deck.splice(0, player.deck.length);
        data.deck.forEach((tower: Tower) => {
            player.deck.push(new Tower(tower.type, tower.speed, tower.damages, tower.area, tower.effect, tower.level));
        });
        for (i = player.board.length; i < data.board.length; i++) {
            board = data.board[i];
            player.board.push(new Board(board.pos, new Tower(board.tower.type, board.tower.speed, board.tower.damages, board.tower.area, board.tower.effect, board.tower.level)));
        }
        player.bullets.splice(0, player.bullets.length);
        data.bullets.forEach((bullet: Bullet) => {
            if (bullet)
                player.bullets.push(new Bullet(bullet.type, bullet.rank, bullet.pos, bullet.target, bullet.travel));
        });
    }

    function updateGame(data: GameTd) {
        gameTd.level = data.level;
        gameTd.timer = data.timer;
        gameTd.start = data.start;
        if (data.state === 1 && gameTd.state === 0.5)
            gameTd.state = 1;
        if (data.state === 2)
            gameTd.state = 2
        gameTd.boss = data.boss;
    }

    try {
        const socketTd = new WebSocket("tower-defense/ws");
        tdConnect = true;
        socketTd.onopen = function () {
            console.log("Connected to TD server");
            socketTd.send(JSON.stringify({event: "socketInit", nick: nick, room: room}));
        };
        socketTd.onmessage = function (event) {
            const data = JSON.parse(event.data);
            switch (data.class) {
                case "gameUpdate":
                    updateGame(data.game);
                    updatePlayer(data.player1);
                    updatePlayer(data.player2);
                    break;
                case "Tower":
                    allTowers.push(new Tower(data.type, data.speed, data.damages, data.area, data.effect, data.level));
                    break;
                case "Disconnected":
                    if (gameTd.state !== 2)
                        gameTd.state = 2.5;
                    break;
                case "Id":
                    player1.id = data.id;
                    break;
                default:
                    console.warn("Unknown type received:", data);
            }
        };

        canvasTd.addEventListener("click", (event: MouseEvent) => {
            const rect = canvasTd.getBoundingClientRect();
            const scaleX = canvasTd.width / rect.width;
            const scaleY = canvasTd.height / rect.height;
            const x = (event.clientX - rect.left) * scaleX;
            const y = (event.clientY - rect.top) * scaleY;
            switch (gameTd.state) {
                case 0:
                    if (x >= 0.1 * canvasTd.width && x < canvasTd.width * 0.9 && y >= 0.46 * canvasTd.height && y < 0.78 * canvasTd.height) {
                        const select = Math.floor((x - 0.1 * canvasTd.width) / (canvasTd.width * 0.16)) + 5 * Math.floor((y - 0.46 * canvasTd.height) / (canvasTd.height * 0.16));
                        if (selected.includes(select)) {
                            const index = selected.indexOf(select);
                            if (index !== -1) {
                                selected.splice(index, 1);
                            }
                        } else if (selected.length < 5)
                            selected.push(select);
                    }
                    if (selected.length === 5 && x >= 0.37 * canvasTd.width && x < 0.63 * canvasTd.width && y >= 0.85 * canvasTd.height && y < 0.95 * canvasTd.height) {
                        socketTd.send(JSON.stringify({
                            event: "towerInit",
                            mode: "default",
                            t1: selected[0],
                            t2: selected[1],
                            t3: selected[2],
                            t4: selected[3],
                            t5: selected[4]
                        }));
                        gameTd.state = 0.5;
                    }
                    if (x >= 0.63 * canvasTd.width && x < 0.66 * canvasTd.width && y >= 0.87 * canvasTd.height && y < 0.93 * canvasTd.height) {
                        selected.splice(0, selected.length);
                        while (selected.length < 5) {
                            const add = Math.floor(Math.random() * 10);
                            if (!selected.includes(add))
                                selected.push(add);
                        }
                    }
                    break;
                case 1:
                    if (y >= canvasTd.height - tile * 1.25 && y < canvasTd.height - tile * 0.25) {
                        if (x >= 0 && x < tile * 5)
                            socketTd.send(JSON.stringify({event: "click", player: 1, button: Math.floor(x / tile)}));
                        else if (x >= tile * 6 && x < tile * 7) {
                            socketTd.send(JSON.stringify({event: "click", player: 1, button: 5}));
                        }
                    }
                    break;
                default:
                    break;
            }
        });

        canvasTd.addEventListener("mousemove", (event) => {
            const rect = canvasTd.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            rdmhover = x >= 0.63 * canvasTd.width && x < 0.66 * canvasTd.width && y >= 0.87 * canvasTd.height && y < 0.93 * canvasTd.height;
        });

        socketTd.onclose = function () {
            return (console.log("Disconnected from TD server"));
        };
        connectionCheck(socketTd);
    } catch (error) {
        console.error("Unexpected error: ", error);
    }
}

export function tdStop() {
    tdConnect = false;
}