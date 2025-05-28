import {Player} from "./api.js";
import {rooms} from "./netcode.js";

export function joinRoomSpec(player: Player, roomId: number) {
    if (roomId !== -1) { // Joining a defined room (invite or tournaments)
        for (const room of rooms) {
            if (room.id === roomId && !room.ended) {
                room.specs.push(player);
                console.log(player.paddle.name, "joined room", room.id, "as spectator");
                return ;
            }
        }
    }
    // Basic random matchmaking
    for (let room of rooms) {
        if (room.players.length === 2 && room.type !== "tournament" && !room.ended) {
            room.specs.push(player);
            console.log(player.paddle.name, "joined room", room.id, "as spectator");
            return ;
        }
    }
    console.log("No room available for spectator");
    if (player.ws.readyState !== WebSocket.CLOSED)
        setTimeout(() => {joinRoomSpec(player, roomId);}, 5000);
}

export function leaveRoomSpec(userId: number) {
    for (let i = 0; i < rooms.length; i++) {
        const room = rooms[i];
        const playerIndex = room.specs.findIndex(player => player.id === userId);
        if (playerIndex !== -1) {
            const player = room.specs[playerIndex];
            console.log(`spectator: ${player.name} with id: ${userId} left room ${room.id}`);
            room.specs.splice(playerIndex, 1);
            return (room.id);
        }
    }
    console.log("Player has not joined a room yet.");
    return (-1);
}

export function changeRoomSpec(player: Player) {
    const roomId = leaveRoomSpec(player.id);
    if (roomId === -1)
        joinRoomSpec(player, -1);
    const roomIndex = rooms.findIndex(room => room.id === roomId);
    const tournament = rooms[roomIndex].type === "tournament";
    if (tournament) {
        for (const room of rooms) {
            if (room.id !== roomId && room.type === "tournament" && !room.ended) {
                room.specs.push(player);
                console.log(player.paddle.name, "joined room", room.id, "as spectator");
                return;
            }
        }
        joinRoomSpec(player, roomId);
    } else {
        for (const room of rooms) {
            if (room.id !== roomId && room.players.length === 2 && !room.ended) {
                room.specs.push(player);
                console.log(player.paddle.name, "joined room", room.id, "as spectator");
                return;
            }
        }
        joinRoomSpec(player, -1);
    }
}