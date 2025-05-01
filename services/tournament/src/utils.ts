import fetch from "undici";
import {UnauthorizedError} from "./error.js";


export const authUser = async (id: number) => {
    const result = await fetch(`http://user-management:5000/authId/${id}`)
    if (!result.ok)
        throw new UnauthorizedError(`this id doesn't exist in database`, `internal server error`)
}

export const fetchNotifyUser = async (ids: number[], event: string, data: any) => {
    const response = await fetch('http://user-management:5000/notify', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ids: ids, event: event, data: data }),
    })
}