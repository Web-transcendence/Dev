// Netcode
import {
    Ball,
    gameState,
    hazardGenerator,
    INTERNAL_PASSWORD,
    moveBall,
    moveHazard,
    movePaddle,
    Player,
    Room,
    timerCheck
} from "./api.js";
import {getWinnerId, insertMatchResult} from "./database.js";
import {fetchNotifyUser} from "./utils.js";

export let rooms: Room[] = [];

function checkId(id: number) {
    for (const room of rooms) {
        if (room.id === id)
            return (false);
    }
    return (true);
}

export function generateRoom() {
    let roomId: number;
    do {
        roomId = Math.floor(Math.random() * 9000 + 1000);
    } while (!checkId(roomId));
    rooms.push(new Room(roomId));
    return (roomId);
}

function checkRoom(room: Room, game: gameState) {
    if (game.state === 2)
        room.ended = true;
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

export function leaveRoom(userId: number) {
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
                const winner = room.players[winnerIndex];
                insertMatchResult(playerA.dbId, playerB.dbId, scoreA, scoreB, winnerIndex);
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
            if (rooms[i].id === roomId) {
                if (rooms[i].players.length === 0) {
                    player.paddle.x = 30;
                    rooms[i].players.push(player);
                    return ;
                } else {
                    player.paddle.x = 1200 - 30;
                    rooms[i].players.push(player);
                    break ;
                }
            }
        }
    } else { // Basic random matchmaking
        for (; i < rooms.length; i++) {
            if (rooms[i].players.length === 1) {
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
    const ball = new Ball (1200 / 2, 800 / 2, 0, 8, 12, "#fcc800");
    const game = new gameState();
    const freq1 = rooms[i].players[0].frequency;
    const freq2 = rooms[i].players[1].frequency
    const intervalId1 = setInterval(() => {
        let i = rooms.findIndex(room => room.id === id);
        if (i === -1) {
            clearInterval(intervalId1);
            return;
        }
        if (rooms[i].players.length === 2) {
            rooms[i].players[0].ws.send(JSON.stringify(rooms[i].players[0].paddle));
            rooms[i].players[0].ws.send(JSON.stringify(rooms[i].players[1].paddle));
            rooms[i].players[0].ws.send(JSON.stringify(ball));
            rooms[i].players[0].ws.send(JSON.stringify(game));
        } else
            clearInterval(intervalId1);
    }, freq1); //Send game info to player 1
    const intervalId2 = setInterval(() => {
        let i = rooms.findIndex(room => room.id === id);
        if (i === -1) {
            clearInterval(intervalId2);
            return;
        }
        if (rooms[i].players.length === 2) {
            rooms[i].players[1].ws.send(JSON.stringify(rooms[i].players[0].paddle));
            rooms[i].players[1].ws.send(JSON.stringify(rooms[i].players[1].paddle));
            rooms[i].players[1].ws.send(JSON.stringify(ball));
            rooms[i].players[1].ws.send(JSON.stringify(game));
        } else
            clearInterval(intervalId2);
    }, freq2); //Send game info to player 2
    const intervalId3 = setInterval(() => {
        let i = rooms.findIndex(room => room.id === id);
        if (i === -1) {
            clearInterval(intervalId3);
            return;
        }
        rooms[i].specs.forEach(spec => {
            if (rooms[i].players.length === 2 && game.state < 2) {
                spec.ws.send(JSON.stringify(rooms[i].players[0].paddle));
                spec.ws.send(JSON.stringify(rooms[i].players[1].paddle));
                spec.ws.send(JSON.stringify(ball));
                spec.ws.send(JSON.stringify(game));
            }
        });
        if (rooms[i].players.length !== 2)
            clearInterval(intervalId3);
    }, 10); //Send game info to spectators
    game.state = 1;
    moveBall(ball, rooms[i].players[0], rooms[i].players[1], game, rooms[i]);
    movePaddle(rooms[i].players[0].input, rooms[i].players[1].input, rooms[i].players[0].paddle, rooms[i].players[1].paddle, game);
    moveHazard(game, ball);
    hazardGenerator(game);
    timerCheck(game);
    checkRoom(rooms[i], game);
}

function isTournamentMatchEnded(roomId: number): boolean {
    const room = rooms.find(room => room.id === roomId);
    return room ? room.ended : true;
}

export async function startInviteMatch(requester: number, opponent: number) {
    const roomId = generateRoom();
    await fetchNotifyUser([opponent], `invitationGame`, roomId);
    return (roomId);
}

export async function waitForMatchEnd(roomId: number, playerA_id: number, playerB_id: number): Promise<number | null> {
    while (true) {
        if (isTournamentMatchEnded(roomId)) {
            return getWinnerId(playerA_id, playerB_id);
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
    }
}

export async function startTournamentMatch(playerA_id: number, playerB_id: number) {
    const roomId = generateRoom();
    await fetchNotifyUser([playerA_id, playerB_id], `invitationGame`, {roomId: roomId})
    waitForMatchEnd(roomId, playerA_id, playerB_id).then(async (winnerId) => {
        if (winnerId)
            await fetch(`http://tournament:7000/userWin/${winnerId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `${INTERNAL_PASSWORD}`
                }
            })
    })
}