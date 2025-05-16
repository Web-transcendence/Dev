import { FastifyReply, FastifyRequest, FastifyInstance } from "fastify"
import * as Schema from "./schema.js"
import {InputError, MyError, ServerError, UnauthorizedError} from "./error.js";
import sanitizeHtml from "sanitize-html"
import {addFriend, checkFriend, getFriendList, removeFriend} from "./friend.js";
import {authUser, fetchId} from "./utils.js";
import {INTERNAL_PASSWORD} from "./api.js";
import {addFriend, getFriendList, removeFriend} from "./friend.js";
import {authUser} from "./utils.js";

const internalVerification = async (req, res) => {
    if (req.headers.authorization !== INTERNAL_PASSWORD)
        throw new UnauthorizedError(`bad internal password to access to this url: ${req.url}`, `internal server error`)
}

export default async function socialRoutes(app: FastifyInstance) {

    app.post('/add', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            console.log('add friend')
            const zod_result = Schema.manageFriendSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(zod_result.error.message, zod_result.error.message)
            let friendNickName = sanitizeHtml(zod_result.data.friendNickName)
            if (!friendNickName)
                throw new InputError(`empty nickname for the friend to had`, `empty nickname`)

            const id: number = Number(req.headers.id)
            if (!id)
                throw new Error("cannot recover id")

            await authUser(id)

            const result = await addFriend(id, friendNickName)

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
                throw new InputError(zod_result.error.message, zod_result.error.message)
            let friendNickName = sanitizeHtml(zod_result.data.friendNickName)
            if (!friendNickName)
                throw new InputError(`Empty nickname to remove from friendList`, `empty nickname`)

            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            await authUser(id)
            await removeFriend(id, friendNickName)

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

    app.post('/checkFriend', {preHandler: internalVerification}, async (req: FastifyRequest, res: FastifyReply) => {
        try {

            const zod_result = Schema.checkFriendSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(`Cannot parse the input`, `id have to be a number`)
            checkFriend(zod_result.data)
            return res.status(200).send()
        } catch (err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })
}
