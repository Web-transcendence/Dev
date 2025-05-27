import {INTERNAL_PASSWORD, Player} from "./api.js";

export const fetchIdByNickName = async (nickName: string): Promise<number> => {
    if (nickName.includes(" "))
        return (-1);
    if (nickName === "IA")
        return (-2);
    const response = await fetch(`http://user-management:5000/idByNickName/${nickName}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': `${INTERNAL_PASSWORD}`
        },
    });
    if(!response.ok) {
        throw new Error(`this nickname doesn't exist`);
    }
    const {id} = await response.json() as {id: number};
    return id;
}

export const fetchMmrById = async (dbId: number): Promise<number> => {
    if (dbId === -1)
        return (1200);
    const response = await fetch(`http://user-management:5000/td/mmrById/${dbId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': `${INTERNAL_PASSWORD}`
        },
    });
    if(!response.ok) {
        throw new Error(`this nickname doesn't exist`);
    }
    const {mmr} = await response.json() as {mmr: number};
    return mmr;
}

export async function updateMmr(playerA: Player, playerB: Player, resultA: number, K = 32) {
    resultA = resultA === 1 ? 0 : 1;
    const expectedA = 1 / (1 + 10 ** ((playerB.mmr - playerA.mmr) / 400));
    const expectedB = 1 - expectedA;
    const resultB = 1 - resultA;

    const newMmrA = Math.max(0, Math.round(playerA.mmr + K * (resultA - expectedA)));
    const newMmrB = Math.max(0, Math.round(playerB.mmr + K * (resultB - expectedB)));

    await Promise.all([
        putNewMmr(playerA.dbId, newMmrA),
        putNewMmr(playerB.dbId, newMmrB)
    ]);
}

export const putNewMmr = async (dbId: number, newMmr: number): Promise<void> => {
    if (dbId === -1) // Si Guest, on ne fait rien
        return ;
    const response = await fetch(`http://user-management:5000/td/mmrById/${dbId}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
            'authorization': `${INTERNAL_PASSWORD}`
        },
        body: JSON.stringify({ mmr: newMmr })
    });
    if (!response.ok) {
        throw new Error(`Failed to update MMR for player with id ${dbId}`);
    }
};

export const fetchNotifyUser = async (ids: number[], event: string, data: any) => {
    const response = await fetch('http://user-management:5000/notify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `${INTERNAL_PASSWORD}`
        },
        body: JSON.stringify({ids: ids, event: event, data: data}),
    })
}

export const fetchPlayerWin = async (winnerId: number) => {
    await fetch(`http://tournament:7000/userWin/${winnerId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `${INTERNAL_PASSWORD}`
        }
    })
}