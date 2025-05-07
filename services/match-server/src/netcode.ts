// Netcode
import {Ball, gameState, hazardGenerator, moveBall, moveHazard, movePaddle, Player, Room, timerCheck} from "./api.js";
import {insertMatchResult} from "./database.js";

let rooms: Room[] = [];

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
    if (game.winner !== "none" || room.players.length !== 2) {
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
    for (let i = 0 ; i < rooms.length; i++) {
        for (let j = 0; j < rooms[i].players.length; j++) {
            if (rooms[i].players[j].id === userId) {
                if (rooms[i].players.length === 2 && Number(rooms[i].players[0].paddle.score) < 6 && Number(rooms[i].players[1].paddle.score) < 6)
                    insertMatchResult(rooms[i].players[0].dbId, rooms[i].players[1].dbId, Number(rooms[i].players[0].paddle.score), Number(rooms[i].players[1].paddle.score), j = 0 ? 1 : 0);
                console.log("player: ", rooms[i].players[j].name, " with id: ", userId, " left room ", rooms[i].id);
                rooms[i].players.splice(j, 1);
                if (rooms[i].players.length === 0) {
                    console.log("room: ", rooms[i].id, " has been cleaned.");
                    rooms.splice(i, 1);
                }
                return ;
            }
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
                break;
            }
        }
        if (id === -1) {
            player.paddle.x = 30;
            let room = new Room(rooms.length);
            room.players.push(player);
            rooms.push(room);
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
    }, freq1);
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
    }, freq2);
    game.state = 1;
    moveBall(ball, rooms[i].players[0], rooms[i].players[1], game);
    movePaddle(rooms[i].players[0].input, rooms[i].players[1].input, rooms[i].players[0].paddle, rooms[i].players[1].paddle, game);
    moveHazard(game, ball);
    hazardGenerator(game);
    timerCheck(game);
    checkRoom(rooms[i], game);
}

function checkTournamentResult(roomId: number) {
    for (const room of rooms) {
        if (room.id === roomId) {
            return true; //Game still in progress
        }
    }
    return false; //Game ended;
}

export function startInviteMatch(requester: number, opponent: number) {
    const roomId = generateRoom();
    //send roomId to usermanagement
}

export function startTournamentMatch(playerA_id: number, playerB_id: number) {
    const roomId = generateRoom();
    //send roomId to usermanagement

}