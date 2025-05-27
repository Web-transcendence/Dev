import {joinRoomTd, Player, roomsTd, RoomTd} from "./api.js";
import {fetchNotifyUser, fetchPlayerWin} from "./utils.js";

export class waitingPlayer {
    player: Player;
    wait: number = 0;
    constructor(player: Player) {
        this.player = player;
    }
}

export const waitingList: waitingPlayer[] = [];
export let matchMakingUp: boolean = false;

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
    await roomWatcher(300, roomId, 0, userId); // change this value to change afk trigger time
    return (roomId);
}

async function roomWatcher(timer: number, roomId: number, clock: number, playerA_id: number) {
    if (clock <= timer) // Time needed to consider the player afk
        setTimeout(() => roomWatcher(timer, roomId, clock + 1, playerA_id), 1000); // Check every second
    else {
        const room = roomsTd.find(room => room.id === roomId);
        if (!room || room.players.length >= 2)
            return;
        else if (room.players.length === 1) {
            if (room.type === "tournament")
                await fetchPlayerWin(room.players[0].dbId); // Inform the tournament service that remaining player won by forfeit
            room.players.forEach(player => {
                player.ws.send(JSON.stringify({ class: "AFK" }));
                player.ws.close();
            });
        } else { // Case where no player joined the room (i.e. double loss)
            if (room.type === "tournament")
                await fetchPlayerWin(playerA_id * -1);
            const i = roomsTd.findIndex(room => room.id === roomId);
            roomsTd.splice(i, 1);
        }
    }
}

export async function startTournamentMatch(playerA_id: number, playerB_id: number) {
    const roomId = generateRoom("tournament");
    await fetchNotifyUser([playerA_id, playerB_id], `invitationTournamentTower`, {roomId: roomId})
    await roomWatcher(60, roomId, 0, playerA_id);
}

function mmrRange(wait: number) {
    return (300 * Math.log2(1 + wait / 60));
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

export async function matchMaking() {
    console.log("Matchmaking service running");
    matchMakingUp = true;
    for (const seeker of waitingList) {
        seeker.wait += 1;
        for (const target of waitingList) {
            if (seeker === target || !canMatch(seeker, target))
                continue;
            const roomId = generateRoom("ranked");
            await joinRoomTd(seeker.player, roomId);
            await joinRoomTd(target.player, roomId);
            removeWaitingPlayer(seeker.player);
            removeWaitingPlayer(target.player);
            break;
        }
    }
    if (waitingList.length !== 0)
        setTimeout(() => matchMaking(), 1000);
    else {
        matchMakingUp = false;
        console.log("Matchmaking service stopped");
    }
}