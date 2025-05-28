import {Player, roomsTd} from "./api.js";


export function joinRoomSpec(player: Player, roomId: number) {
    if (roomId !== -1) { // Joining a defined room (invite or tournaments)
        for (const room of roomsTd) {
            if (room.id === roomId && !room.ended) {
                room.specs.push(player);
                console.log(player.name, "joined room", room.id, "as spectator");
                return ;
            }
        }
    }
    // Basic random matchmaking
    for (let room of roomsTd) {
        if (room.players.length === 2 && !room.ended) {
            room.specs.push(player);
            console.log(player.name, "joined room", room.id, "as spectator");
            return ;
        }
    }
    console.log("No room available for spectator");
    if (player.ws.readyState !== WebSocket.CLOSED)
        setTimeout(() => {joinRoomSpec(player, roomId);}, 5000);
}

export function leaveRoomSpec(userId: number) {
    for (let i = 0; i < roomsTd.length; i++) {
        const room = roomsTd[i];
        const playerIndex = room.specs.findIndex(player => player.id === userId);
        if (playerIndex !== -1) {
            const player = room.specs[playerIndex];
            console.log(`spectator: ${player.name} with id: ${userId} left room ${room.id}`);
            room.specs.splice(playerIndex, 1);
            return (room.id);
        }
    }
    console.log("Player has not joined a room yet.");
}

export function changeRoomSpec(player: Player) {
    const roomId = leaveRoomSpec(player.id);
    for (const room of roomsTd) {
        if (room.id !== roomId && room.players.length === 2 && !room.ended) {
            room.specs.push(player);
            console.log(player.name, "joined room", room.id, "as spectator");
            return ;
        }
    }
    joinRoomSpec(player, -1);
}