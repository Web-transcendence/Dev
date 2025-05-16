import {displayNotification} from "./notificationHandler.js";

export const fetchInvitation = async (game: string, id: number) => {
    const token = sessionStorage.getItem('token');
    console.log(token)
    const response = await fetch(`${game}/invitationGame/${id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': 'Bearer ' + token
        }
    });
    if (!response.ok) {
        const error = await response.json();
        console.log(error.message);
        displayNotification(`Can't create game`, {type: "error"})
        return ;
    }
    const {roomId} = await response.json()
    return roomId
}

export const fetchRefuseInvitation = async (game: string, id: number) => {
    const token = sessionStorage.getItem('token');
    console.log(token)
    const response = await fetch(`${game}/invitationGame/${id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': 'Bearer ' + token
        }
    });
    if (!response.ok) {
        const error = await response.json();
        console.log(error.message);
        displayNotification(`Can't create game`, {type: "error"})
        return ;
    }
    const {roomId} = await response.json()
    return roomId
}