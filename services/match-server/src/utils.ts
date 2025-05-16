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

export const fetchNickNameById = async (id: number): Promise<string> => {
    const response = await fetch(`http://user-management:5000/nickNameById/${id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': `${INTERNAL_PASSWORD}`
        },
    });
    if(!response.ok) {
        throw new Error(`this id doesn't exist`);
    }
    const {nickName} = await response.json() as {nickName: string};
    return nickName;
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