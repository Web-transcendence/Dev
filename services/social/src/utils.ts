import {fetch} from "undici";
import {NotFoundError, ServerError, UnauthorizedError} from "./error.js";
import {INTERNAL_PASSWORD} from "./api.js";


export async function authUser(id: number) {
    const result = await fetch(`http://user-management:5000/authId/${id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': `${INTERNAL_PASSWORD}`
        },
    })
    if (!result.ok)
        throw new UnauthorizedError(`this id doesn't exist in database`, `internal server error`)
}

export async function fetchId(nickName: string) {
    const result = await fetch(`http://user-management:5000/idByNickName/${nickName}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'authorization': `${INTERNAL_PASSWORD}`
        },
    })
    if (!result.ok)
        throw new NotFoundError(`fetchId`, 'user not found')
    const { id } = await result.json()
    return id
}

export const fetchNotifyUser = async (ids: number[], event: string, data: any) => {
    console.log(`notify : ${event}`)
    const response = await fetch('http://user-management:5000/notify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'authorization': `${INTERNAL_PASSWORD}`
        },
        body: JSON.stringify({ids: ids, event: event, data: data }),
    })
    if (!response.ok)
        throw new ServerError(`fetch notify error`, 500)
}