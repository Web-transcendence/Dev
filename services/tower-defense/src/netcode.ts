import {roomsTd, RoomTd} from "./api.js";

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