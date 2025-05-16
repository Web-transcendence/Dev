import { FastifyReply, FastifyRequest, FastifyInstance } from "fastify"
import {INTERNAL_PASSWORD, tournamentSessions} from './api.js';
import * as Schema from "./schema.js"
import {ConflictError, InputError, MyError, ServerError, UnauthorizedError} from "./error.js";
import {authUser} from "./utils.js";


const internalVerification = async (req, res) => {
    if (req.headers.authorization !== INTERNAL_PASSWORD)
        throw new UnauthorizedError(`bad internal password to access to this url: ${req.url}`, `internal server error`)
}

export default async function tournamentRoutes(app: FastifyInstance) {

    app.post('/join', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const zod_result = Schema.tournamentIdSchema.safeParse(req.body)
            if (!zod_result.success)
                throw new InputError(`this id of tournament doesn't exist`, `the tournament id have to be a number`)
            let idTournament = zod_result.data.tournamentId
            console.log('id is tournament', idTournament);

            if (!idTournament)
                throw new InputError(`Empty tournament id`, `tournamentId is empty`)

            const id: number = Number(req.headers.id)
            console.log('id is user', id);
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            await authUser(id);

            const tournament = tournamentSessions.get(idTournament)
            if (!tournament)
                throw new ConflictError(`there is no tournament with this id`, `this id of tournament doesn't exist`)

            await tournament.addParticipant(id)

            return res.status(200).send()
        }
        catch(err) {
            if (err instanceof MyError) {
                console.error(err.message)
                console.log(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            console.log(err)
            return res.status(500).send()
        }
    })

    app.get('/getList', (req: FastifyRequest, res: FastifyReply) => {
        const tournamentList: {participants: number[], maxPlayer: number, status: string}[] = []

        for (const [id, tournament] of tournamentSessions)
            tournamentList.push(tournament.getData())

        return res.status(200).send(tournamentList)
    })

    app.get('/quit', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            await authUser(id);
            console.log('qqqqqqqqqqqqqq')
            for (const [tournamentId, tournament] of tournamentSessions)
                if (tournament.hasParticipant(id)) {
                    console.log('quit tournament')
                    await tournament.quit(id)
                }

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

    app.get('/launch', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            console.log('launch')
            const id: number = Number(req.headers.id)
            if (!id)
                throw new ServerError(`cannot parse id, which should not happen`, 500)

            await authUser(id);

            let tournamentId: number = 0
            for (const [tempId, tournament] of tournamentSessions) {
                if (tournament.hasParticipant(id))
                    tournamentId = tempId
            }

            if (!tournamentId)
                throw new ConflictError(`there is no tournament with this id`, `you are not in a tournament`)

            const tournament = tournamentSessions.get(tournamentId)
            if (!tournament)
                throw new ConflictError(`there is no tournament with this id`, `internal error system`)

            tournament.launch().then(() => {
                console.log('launch done');
            });

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

    // app.get(`/RunningTournamentInformation/:id`, async (req: FastifyRequest, res: FastifyReply) => {
    //     try {
    //         const {id} = req.params as { id: string }
    //
    //         const numericId = Number(id)
    //
    //         if (isNaN(numericId))
    //             throw new InputError(`the id isn't a number`, `id have to be a number`)
    //
    //         const tournament = tournamentSessions.get(numericId)
    //         if (!tournament)
    //             throw new InputError(`this id of tournament doesn't exist`, `only these id of tournament exist : 4,8,16,32`)
    //
    //         return res.status(200).send(tournament.sessionData())
    //
    //     } catch (err) {
    //         if (err instanceof MyError) {
    //             console.error(err.message)
    //             return res.status(err.code).send({error: err.toSend})
    //         }
    //         console.error(err)
    //         return res.status(500).send()
    //     }
    // })

    app.get(`/userWin/:id`, {preHandler: internalVerification} , async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const {id} = req.params as { id: string }

            const numericId = Number(id)

            if (isNaN(numericId))
                throw new InputError(`the id isn't a number`, `id have to be a number`)

            // tournamentSession.

            for (const [tournamentId, tournament] of tournamentSessions) {
                if (tournament.hasPlayer(numericId)) {
                    await tournament.bracketWon(numericId)
                    return res.status(200).send()
                }
            }
            return res.status(404).send()
        } catch (err) {
            if (err instanceof MyError) {
                console.error(err.message)
                return res.status(err.code).send({error: err.toSend})
            }
            console.error(err)
            return res.status(500).send()
        }
    })

    app.get(`/test`, async (req: FastifyRequest, res: FastifyReply) => {
        for (const [id, tournament] of tournamentSessions) {
            console.log(`tournament player:`, tournament.getData())
        }
        return res.status(200).send()
    })

}
