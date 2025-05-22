import {roomsTd, RoomTd} from "./api.js";
import {fetchNotifyUser, fetchPlayerWin} from "./utils.js";
import {getWinnerId} from "./database.js";

function checkId(id: number) {
    for (const roomTd of roomsTd) {
        if (roomTd.id === id)
            return false;
    }
    return true;
}

export function generateRoom(mode?: string) {
    let roomId: number;
    do {
        roomId = Math.floor(Math.random() * 9000 + 1000);
    } while (!checkId(roomId));
    roomsTd.push(new RoomTd(roomId, mode));
    return (roomId);
}

export async function startInviteMatch(userId: number, opponent: number) {
    const roomId = generateRoom();

    await fetchNotifyUser([opponent], `invitationTowerDefense`, {roomId: roomId, id: userId});
    return (roomId);
}

async function roomWatcher(roomId: number, clock: number, playerA_id: number) {
    if (clock <= 60) // Time needed to consider the player afk
        setTimeout(() => roomWatcher(roomId, clock + 1, playerA_id), 1000); // Check every second
    else {
        const room = roomsTd.find(room => room.id === roomId);
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
            const i = roomsTd.findIndex(room => room.id === roomId);
            roomsTd.splice(i, 1);
        }
    }
}

export async function startTournamentMatch(playerA_id: number, playerB_id: number) {
    const roomId = generateRoom("tournament");
    await fetchNotifyUser([playerA_id, playerB_id], `invitationTournamentTower`, {roomId: roomId})
    await roomWatcher(roomId, 0, playerA_id);
}