import { FastifyReply, FastifyRequest, FastifyInstance } from "fastify"
import sanitizeHtml from "sanitize-html";
import {tournamentSessions} from './api.js';
import * as Schema from "./schema.js"
import fetch from 'undici'
import {ConflictError, InputError, MyError, ServerError, UnauthorizedError} from "./error.js";

async function authUser(id: number) {
    const result = await fetch(`http://user-management:5000/authId/${id}`)
    if (!result.ok)
        throw new UnauthorizedError(`this id doesn't exist in database`, `internal server error`)
}

export default async function tournamentRoutes(app: FastifyInstance) {

    app.post('/join', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.tournamentIdSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(`Cannot parse the input`)
            let idTournament = Number(sanitizeHtml(zod_result.data.tournamentId))
            if (!idTournament)
                throw new InputError(`Empty tournament input`)

            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            authUser(id);

            const tournament = tournamentSessions.get(idTournament)
            if (!tournament)
                throw new ConflictError(`there is no tournament with this id`, 'internal error system')

            tournament.addParticipant(id)

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

    app.get('/getList', (req: FastifyRequest, res: FastifyReply) => {
        const tournamentList: {participants: number[], maxPlayer: number, status: string}[] = []

        for (const [id, tournament] of tournamentSessions)
            tournamentList.push(tournament.getData())

        return res.status(200).send(tournamentList)
    })

    app.post('/quit', (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            authUser(id);

            for (const [id, tournament] of tournamentSessions)
                if (tournament.hasParticipant(id))
                    tournament.quit(id)

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

    app.post('/launch', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            authUser(id);

            const tournament = tournamentSessions.get(id)
            if (!tournament)
                throw new ConflictError(`there is no tournament with this id`, 'internal error system')

            const result = await tournament.launch()

            return res.status(200).send({result: result})
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
