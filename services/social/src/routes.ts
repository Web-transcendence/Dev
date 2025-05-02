import { FastifyReply, FastifyRequest, FastifyInstance } from "fastify"
import * as Schema from "./schema.js"
import {InputError, MyError, NotFoundError, ServerError, UnauthorizedError} from "./error.js";
import sanitizeHtml from "sanitize-html"
import {addFriend, getFriendList, removeFriend} from "./friend.js";
import {fetch} from 'undici'

async function authUser(id: number) {
    const result = await fetch(`http://user-management:5000/authId/${id}`)
    if (!result.ok)
        throw new UnauthorizedError(`this id doesn't exist in database`, `internal server error`)
}

export async function fetchId(nickName: string) {
    const result = await fetch(`http://user-management:5000/idByNickName/${nickName}`)
    if (!result.ok)
        throw new NotFoundError(`fetchId`, 'user not found')
    const { id } = await result.json()
    return id
}

export default async function socialRoutes(app: FastifyInstance) {

    app.post('/add', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.manageFriendSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(`Cannot parse the input`)
            let friendNickName = sanitizeHtml(zod_result.data.friendNickName)
            if (!friendNickName)
                throw new InputError(`Empty nickname to add`)

            const id: number = Number(req.headers.id)
            if (!id)
                throw new Error("cannot recover id")

            await authUser(id)

            const result = await addFriend(id, friendNickName)
            //sse

            return res.status(200).send({message: result})
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })

    app.get('/list', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            await authUser(id);

            const result = getFriendList(id)

            return res.status(200).send(result)
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })

    app.post('/remove', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.manageFriendSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(`Cannot parse the input`)
            let friendNickName = sanitizeHtml(zod_result.data.friendNickName)
            if (!friendNickName)
                throw new InputError(`Empty nickname to remove from friendList`)

            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            await authUser(id)
            await removeFriend(id, friendNickName)

            //sse
            return res.status(200).send()
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })


}
