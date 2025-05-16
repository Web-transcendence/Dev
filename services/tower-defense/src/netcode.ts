import {roomsTd, RoomTd} from "./api.js";
import {fetchNotifyUser} from "./utils.js";
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

function isTournamentMatchEnded(roomId: number): boolean {
    const room = roomsTd.find(room => room.id === roomId);
    return room ? room.ended : true;
}

export async function startInviteMatch(userId: number, opponent: number) {
    const roomId = generateRoom();

    await fetchNotifyUser([opponent], `invitationTowerDefense`, {roomId: roomId, id: userId});
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
    await fetchNotifyUser([playerA_id, playerB_id], `invitationGame`, roomId)
    const winnerId = await waitForMatchEnd(roomId, playerA_id, playerB_id);
    return (winnerId);
}