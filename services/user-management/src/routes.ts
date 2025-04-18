import * as Schema from "./schema.js"
import sanitizeHtml from "sanitize-html"
import {User} from "./User.js"
import { FastifyReply, FastifyRequest, FastifyInstance } from "fastify"
import {connectedUsers, tournamentSessions} from "./api.js"
import {EventMessage} from "fastify-sse-v2"
import {tournament} from "./tournament.js"
import {ConflictError, InputError, MyError, ServerError} from "./error.js";




export default async function userRoutes(app: FastifyInstance) {

    app.post('/register', async (req, res) => {
        try {
            const zod_result = Schema.signUpSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(`Cannot parse the input`)

            let {nickName, email, password} = {nickName: sanitizeHtml(zod_result.data.nickName), email: sanitizeHtml(zod_result.data.email), password: sanitizeHtml(zod_result.data.password)}
            if (!nickName || !email || !password)
                throw new InputError(`Empty user data`)

            const token = await User.addClient(nickName, email, password)

            return res.status(201).send({token: token, redirect: "post/login"})
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })


    app.post('/login', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.signInSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(`Cannot parse the input`)

            let {nickName, password} = {
                nickName: sanitizeHtml(zod_result.data.nickName),
                password: sanitizeHtml(zod_result.data.password)
            }
            if (!nickName || !password)
                throw new InputError(`Empty user data`)

            const token = await User.login(nickName, password)
            return res.status(200).send({token: token})
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
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
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })

    app.post("/2faVerify", (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.verifySchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(`Cannot parse the input`)
            let {secret, nickName} = {
                secret : sanitizeHtml(zod_result.data.secret),
                nickName: sanitizeHtml(zod_result.data.nickName),
            }
            if (!secret || nickName)
                throw new InputError(`empty userData for 2fa`)

            const id = User.getIdbyNickName(nickName)

            const user = new User(id)

            const jwt = user.verify(secret)

            return res.status(200).send({token: jwt, nickName: nickName})
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })

    app.get('/getProfile', async (req, res) => {
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
                return res.status(err.code).send({error: err.message})
            }
            console.log(err)
            return res.status(500).send(err)
        }
    })


    app.post('/addFriend', (req: FastifyRequest, res: FastifyReply) => {
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
            const user = new User(id)
            const result = user.addFriend(friendNickName)

            return res.status(200).send({message: result})
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })

    app.get('/friendList', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            const user = new User(id)
            const result = user.getFriendList()

            return res.status(200).send(result)
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })

    app.post('/removeFriend', (req: FastifyRequest, res: FastifyReply) => {
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

            const user = new User(id)
            user.removeFriend(friendNickName)

            return res.status(200).send()
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })


    app.post('/createTournament', (req: FastifyRequest, res: FastifyReply) => {
        try
        {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            const user = new User(id)

            const tournament = user.createTournament()

            tournamentSessions.set(id, tournament)

            return res.status(200).send()
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })

    app.post('/joinTournament', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.manageFriendSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(`Cannot parse the input`)
            let friendNickName = sanitizeHtml(zod_result.data.friendNickName)
            if (!friendNickName)
                throw new InputError(`Empty nickname to add in a tournament`)

            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            new User(id)

            const idToJoin: number= User.getIdbyNickName(friendNickName)

            const tournament = tournamentSessions.get(idToJoin)
            if (!tournament)
                throw new ConflictError(`there is no tournament with this id`)

            tournament.addParticipant(id)

            return res.status(200).send()
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })

    app.get('/getTournamentList', (req: FastifyRequest, res: FastifyReply) => {
        const tournamentList: {creatorId: number, participantCount: number, status: string}[] = []

        for (const [id, tournament] of tournamentSessions)
            tournamentList.push(tournament.getData())

        return res.status(200).send(tournamentList)
    })

    app.post('/quitTournament', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            const user = new User(id)

            const tournament = user.getActualTournament()
            if (!tournament)
                throw new ConflictError(`this user isn't in tournament actually`)

            tournament.quit(id)

            return res.status(200).send()
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })

    app.post('/launchTournament', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            new User(id)

            const tournament = tournamentSessions.get(id)
            if (!tournament)
                throw new ConflictError(`there is no tournament with this id`)

            const result = await tournament.launch()

            return res.status(200).send({result: result})
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.message})
            }
            console.error(err)
            return res.status(500).send(err)
        }
    })

    /**
     * initiate the sse connection between the server and the client, stock the response in a map.
     *      the response can call the method .sse to send data in this format : {data: JSON.stringify({ event: string, data: any })}
     */
    app.get('/sse', async function (req, res) {
        const id: number = Number(req.headers.id)
        if (!id)
            return res.status(500).send({error: "Server error: Id not found"})
        connectedUsers.set(id, res)
        const message: EventMessage = { event: "initiation", data: "Some message" }
        res.sse({data: JSON.stringify(message)})
    })

}