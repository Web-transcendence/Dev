import {INTERNAL_PASSWORD} from "./api.js";

export const fetchIdByNickName = async (nickName: string): Promise<number> => {
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
    const response = await fetch(`http://user-management:5000/mmrById/${dbId}`, {
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

export const fetchNotifyUser = async (ids: number[], event: string, data: any) => {
    const response = await fetch('http://user-management:5000/notify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `${INTERNAL_PASSWORD}`
        },
        body: JSON.stringify({ids: ids, event: event, data: data }),
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