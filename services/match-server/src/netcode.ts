// Netcode
import {
    Ball,
    gameState, generateId,
    hazardGenerator,
    INTERNAL_PASSWORD,
    moveBall,
    moveHazard,
    movePaddle,
    Player,
    Room,
    timerCheck
} from "./api.js";
import {insertMatchResult} from "./database.js";
import {fetchNotifyUser, fetchPlayerWin, updateMmrById} from "./utils.js";

export class waitingPlayer {
    player: Player;
    wait: number = 0;
    constructor(player: Player) {
        this.player = player;
    }
}

export const waitingList: waitingPlayer[] = [];
export let rooms: Room[] = [];

export function checkId(id: number) {
    for (const room of rooms) {
        if (room.id === id)
            return (false);
    }
    return (true);
}

export function generateRoom(mode?: string) {
    let roomId: number;
    do {
        roomId = Math.floor(Math.random() * 9000 + 1000);
    } while (!checkId(roomId));
    rooms.push(new Room(roomId, mode));
    return (roomId);
}

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

export async function leaveRoom(userId: number) {
    for (let i = 0; i < rooms.length; i++) {
        const room = rooms[i];
        const playerIndex = room.players.findIndex(player => player.id === userId);
        if (playerIndex !== -1) {
            const player = room.players[playerIndex];
            console.log(`player: ${player.name} with id: ${userId} left room ${room.id}`);
            if (room.players.length === 2 && Number(room.players[0].paddle.score) < 6 && Number(room.players[1].paddle.score) < 6) {
                const [playerA, playerB] = room.players;
                const scoreA = Number(playerA.paddle.score);
                const scoreB = Number(playerB.paddle.score);
                const winnerIndex = room.players.findIndex(player => player.id !== userId);
                console.log(`winner: ${winnerIndex}`);
                const winner = room.players[winnerIndex];
                if (room.type === "tournament")
                    await fetchPlayerWin(winner.dbId);
                if (room.type === "ranked") {
                    await updateMmrById(playerA.dbId, playerA.mmr, winnerIndex === 0);
                    await updateMmrById(playerB.dbId, playerB.mmr, winnerIndex === 1);
                }
                console.log(`A: ${playerA.dbId} B: ${playerB.dbId} Windex: ${winnerIndex}`);
                insertMatchResult(playerA.dbId, playerB.dbId, scoreA, scoreB, winnerIndex);
                room.ended = true;
                room.specs.forEach(spec => {
                    spec.ws.send(JSON.stringify({ type: "gameEnd", winner: winner.name }));
                });
            }
            room.players.splice(playerIndex, 1);
            if (room.players.length === 0) {
                console.log(`room: ${room.id} has been cleaned.`);
                rooms.splice(i, 1);
            }
            return;
        }
    }
    console.log("Player has not joined a room yet.");
}

export function joinRoom(player: Player, roomId: number) {
    let id: number = -1;
    let i : number = 0;
    if (roomId !== -1) { // Joining a defined room (invite or tournaments)
        for (; i < rooms.length; i++) {
            if (rooms[i].id === roomId && rooms[i].players.length < 2) {
                if (rooms[i].players.length === 0)
                    player.paddle.x = 30;
                else
                    player.paddle.x = 1200 - 30;
                rooms[i].players.push(player);
                id = rooms[i].id;
                console.log(player.paddle.name, "joined room", rooms[i].id);
                break ;
            }
        }
    } else { // Basic random matchmaking
        for (; i < rooms.length; i++) {
            if (rooms[i].id < 1000 && rooms[i].players.length < 2 && rooms[i].players.length === 1) {
                player.paddle.x = 1200 - 30;
                rooms[i].players.push(player);
                id = rooms[i].id;
                console.log(player.paddle.name, "joined room", rooms[i].id);
                break;
            }
        }
        if (id === -1) {
            player.paddle.x = 30;
            let room = new Room(rooms.length);
            room.players.push(player);
            rooms.push(room);
            console.log(player.paddle.name, "created and joined room", rooms[i].id);
            return;
        }
    }
    if (i === rooms.length || rooms[i].players.length !== 2)
        return ;
    console.log(`room ${rooms[i].id}'s game has started`);
    roomLoop(rooms[i]);
}

function roomLoop(room: Room) {
    const ball = new Ball (1200 / 2, 800 / 2, 0, 8, 12, "#fcc800");
    const game = new gameState();
    const freq1 = room.players[0].frequency;
    const freq2 = room.players[1].frequency
    const intervalId1 = setInterval(() => {
        if (room.players.length === 2) {
            const payload = {
                type: "gameUpdate",
                paddle1: room.players[0].paddle,
                paddle2: room.players[1].paddle,
                ball: ball,
                game: game
            };
            room.players[0].ws.send(JSON.stringify(payload));
        } else
            clearInterval(intervalId1);
    }, freq1); //Send game info to player 1
    const intervalId2 = setInterval(() => {
        if (room.players.length === 2) {
            const payload = {
                type: "gameUpdate",
                paddle1: room.players[0].paddle,
                paddle2: room.players[1].paddle,
                ball: ball,
                game: game
            };
            room.players[1].ws.send(JSON.stringify(payload));
        } else
            clearInterval(intervalId2);
    }, freq2); //Send game info to player 2
    const intervalId3 = setInterval(() => {
        if (room.players.length === 2) {
            const payload = {
                type: "gameUpdate",
                paddle1: room.players[0].paddle,
                paddle2: room.players[1].paddle,
                ball: ball,
                game: game
            };
            room.specs.forEach(spec => {
                if (room.players.length === 2 && game.state < 2) {
                    spec.ws.send(JSON.stringify(payload));
                }
            });
        } else
            clearInterval(intervalId3);
    }, 10); //Send game info to spectators
    game.state = 1;
    moveBall(ball, room.players[0], room.players[1], game, room);
    movePaddle(room.players[0].input, room.players[1].input, room.players[0].paddle, room.players[1].paddle, game);
    moveHazard(game, ball);
    hazardGenerator(game);
    timerCheck(game);
    checkRoom(room, game);
}

export async function startInviteMatch(userId: number, opponent: number) {
    const roomId = generateRoom();

    await fetchNotifyUser([opponent], `invitationPong`, {roomId: roomId, id: userId});
    return (roomId);
}

async function roomWatcher(roomId: number, clock: number, playerA_id: number) {
    if (clock <= 60) // Time needed to consider the player afk
        setTimeout(() => roomWatcher(roomId, clock + 1, playerA_id), 1000); // Check every second
    else {
        const room = rooms.find(room => room.id === roomId);
        if (!room || room.players.length >= 2)
            return;
        else if (room.players.length === 1) {
            await fetchPlayerWin(room.players[0].dbId); // Inform the tournament service that remaining player won by forfeit
            room.players.forEach(player => {
                player.ws.send(JSON.stringify({ type: "Disconnected" }));
                player.ws.close();
            });
        } else { // Case where no player joined the room (i.e. double loss)
            await fetchPlayerWin(playerA_id * -1);
            const i = rooms.findIndex(room => room.id === roomId);
            rooms.splice(i, 1);
        }
    }
}

export async function startTournamentMatch(playerA_id: number, playerB_id: number) {
    const roomId = generateRoom("tournament");
    await fetchNotifyUser([playerA_id, playerB_id], `invitationTournamentPong`, {roomId: roomId})
    await roomWatcher(roomId, 0, playerA_id);
}

function mmrRange(wait: number) {
    return (150 * Math.log2(1 + wait / 600));
}

function canMatch(seeker: waitingPlayer, target: waitingPlayer): boolean {
    if (target.player.mmr < seeker.player.mmr - mmrRange(seeker.wait) || target.player.mmr > seeker.player.mmr + mmrRange(seeker.wait)) // Check if target mmr is in seeker's range
        return false;
    if (seeker.player.mmr < target.player.mmr - mmrRange(target.wait) || seeker.player.mmr > target.player.mmr + mmrRange(target.wait)) // Reverse check
        return false;
    return true; // Players can be matched !
}

export function removeWaitingPlayer(player: Player) {
    const index = waitingList.findIndex(wp => wp.player === player);
    if (index !== -1) {
        waitingList.splice(index, 1);
    }
}

function matchMaking() {
    for (const seeker of waitingList) {
        seeker.wait += 1;
        for (const target of waitingList) {
            if (seeker === target || !canMatch(seeker, target))
                continue;
            const roomId = generateRoom("ranked");
            joinRoom(seeker.player, roomId);
            joinRoom(target.player, roomId);
            removeWaitingPlayer(seeker.player);
            removeWaitingPlayer(target.player);
            break ;
        }
    }
    setTimeout(() => matchMaking() ,100);
}

matchMaking();