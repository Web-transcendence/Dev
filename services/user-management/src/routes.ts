import * as Schema from "./schema.js"
import sanitizeHtml from "sanitize-html"
import {User} from "./User.js"
import { FastifyReply, FastifyRequest, FastifyInstance } from "fastify"
import {connectedUsers, INTERNAL_PASSWORD} from "./api.js"
import {InputError, MyError, ServerError, UnauthorizedError} from "./error.js";
import {notifyUser} from "./serverSentEvent.js";
import {nickNameSchema, passwordSchema} from "./schema.js";


export function logConnectedUser() {
    const connected = []
    for (const [id, user] of connectedUsers)
        connected.push(id)
    console.log(connected);
}

const internalVerification = async (req, res) => {
    if (req.headers.authorization !== INTERNAL_PASSWORD)
        throw new UnauthorizedError(`bad internal password to access to this url: ${req.url}`, `internal server error`)
}

export default async function userRoutes(app: FastifyInstance) {

    app.post('/register', async (req, res) => {
        try {
            const zod_result = Schema.signUpSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(zod_result.error.message, zod_result.error.message)

            let {nickName, email, password} = {nickName: sanitizeHtml(zod_result.data.nickName), email: sanitizeHtml(zod_result.data.email), password: sanitizeHtml(zod_result.data.password)}
            if (!nickName || !email || !password)
                throw new InputError(`Empty user data`, `empty value`)

            const token = await User.addClient(nickName, email, password)

            return res.status(201).send({token: token, nickName: nickName, redirect: "post/login"})
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


    app.post('/login', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.signInSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(zod_result.error.message, zod_result.error.message)

            const {nickName, password} = {
                nickName: sanitizeHtml(zod_result.data.nickName),
                password: sanitizeHtml(zod_result.data.password)
            }
            if (!nickName || !password)
                throw new InputError(`Empty user data`, `empty value`)

            const token = await User.login(nickName, password)
            return res.status(200).send({token: token, nickName: nickName})
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

    app.get("/2faInit", async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)
            const user = new User(id)

            const QRCode = await user.generateSecretKey()
            return res.status(200).send(QRCode)
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

    app.post("/2faVerify", (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.verifySchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(zod_result.error.message, zod_result.error.message)
            let {secret, nickName} = {
                secret : sanitizeHtml(zod_result.data.secret),
                nickName: sanitizeHtml(zod_result.data.nickName),
            }
            if (!secret || !nickName)
                throw new InputError(`Empty user data`, `empty value`)
            const id = User.getIdbyNickName(nickName)

            const user = new User(id)

            const jwt = user.verify(secret)
            return res.status(200).send({token: jwt, nickName: nickName})
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

    app.get('/privateProfile', async (req, res) => {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)
            const user = new User(id)

            const profileData = user.getProfile()
            return res.status(200).send(profileData)
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.log(err)
            return res.status(500).send()
        }
    })

    app.post('/userInformation', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.idArraySchema.safeParse(req.body)
            if (!zod_result.success) {
                throw new InputError(`an array of ids is needed to fetch theses informations`, `error 400`)
            }
            const ids: number[] = zod_result.data.ids

            const idsInformation = []

            for (const id of ids) {
                const user = new User(id)
                idsInformation.push(user.publicData())
            }

            return res.status(200).send({usersData: idsInformation})
        } catch (err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.log(err)
            return res.status(500).send()
        }
    })

    /**
     * initiate the sse connection between the server and the client, stock the response in a map.
     *      the response can call the method .sse to send data in this format : {data: JSON.stringify({ event: string, data: any })}
     */
    app.get('/sse', async function (req: FastifyRequest, res: FastifyReply) {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)
            const user = new User(id)
            await user.sseHandler(req, res)
        } catch (err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }

    })

    app.post('/updatePicture', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.pictureSchema.safeParse(req.body);
            if (!zod_result.success)
                throw new InputError(`bad format of picture to update`, zod_result.error.message)
            let pictureURL = sanitizeHtml(zod_result.data.pictureURL);
            if (!pictureURL)
                throw new InputError(`empty url to update`, `empty picture`)

            const id = Number(req.headers.id);
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            const user = new User(id);

            user.updatePictureProfile(pictureURL);
            return res.status(200).send();
        } catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })

    app.get('/getPicture', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id = Number(req.headers.id);
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            const user = new User(id);
            const result = user.getPictureProfile();

            return res.status(200).send({url: result});
        } catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })

    app.post('/setPassword', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = passwordSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(zod_result.error.message, zod_result.error.message)
            const newPassword = zod_result.data.password

            const id = Number(req.headers.id);
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            const user = new User(id);
            user.setPassword(newPassword);

            return res.status(200).send();
        } catch (err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })

    app.post('/setNickName', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = nickNameSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(zod_result.error.message, zod_result.error.message)
            const newNickName = sanitizeHtml(zod_result.data.nickName)
            if (!newNickName)
                throw new InputError('empty nickname given to set', `empty nickname`)
            const id = Number(req.headers.id);
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            const user = new User(id);
            user.setNickname(newNickName);

            return res.status(200).send();
        } catch (err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })

    app.get('/authId/:id', {preHandler: internalVerification} ,  (req: FastifyRequest, res: FastifyReply) => {
        try {
            const { id } = req.params as { id: string }

            const numericId = Number(id)

            if (isNaN(numericId))
                throw new InputError(`id isn't a number`, `the given id is not a number`)

            new User(numericId)

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

    app.get('/idByNickName/:nickName', {preHandler: internalVerification} , (req: FastifyRequest, res: FastifyReply) => {
        try {
            const { nickName }  = req.params as { nickName: string }
            if (!nickName)
                throw new InputError('empty nickname in the param of the idByNickName', `empty nickname`)

            const id = User.getIdbyNickName(nickName)

            return res.status(200).send({id})
        } catch (err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })

    app.post('/notify', {preHandler: internalVerification}, (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.notifySchema.safeParse(req.body);
            if (!zod_result.success)
                throw new InputError(`Bad format to notify client`, `internal error system`)

            const data = zod_result.data

            notifyUser(data.ids, data.event, data.data);

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

    app.get('/invitationGame/:id', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            const friendId = Number((req.params as {id: string}).id)

            const response = fetch(`http://social:6500/checkFriend`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'authorization': `${INTERNAL_PASSWORD}`
                },
                body: JSON.stringify({id1: id, id2: friendId})
            })

            notifyUser([friendId], `invitationGame`, )

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

