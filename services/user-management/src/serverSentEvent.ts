import {FastifyReply} from "fastify";
import {connectedUsers} from "./api.js";
import {fetchAcceptedFriends} from "./utils.js";
import {fetch} from "undici";


export const notifyUser = (ids: number[], event: string, data: any): void => {
    for (const id of ids) {
        const connection: FastifyReply | undefined = connectedUsers.get(id);
        const stringData: string = JSON.stringify(data);
        if (connection)
            connection.sse({event: event, data: stringData});
    }
}


export const disconnect = async (id: number ) => {
    const friends = await fetchAcceptedFriends(id)
    notifyUser(friends, 'disconnection', {id: id})

    await fetch(`http://tournament:7000/quit`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'id': `${id}`
        }
    })
    // fetch game disconnection
}

export const connection = async (id: number ) => {
    const friends = await fetchAcceptedFriends(id)
    notifyUser(friends, 'connection', {id: id})
}