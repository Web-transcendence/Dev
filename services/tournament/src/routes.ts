import { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify'
import { INTERNAL_PASSWORD, tournamentSessions } from './api.js'
import * as Schema from './schema.js'
import { ConflictError, InputError, MyError, ServerError, UnauthorizedError } from './error.js'
import { authUser } from './utils.js'


const internalVerification = async (req, res) => {
	if (req.headers.authorization !== INTERNAL_PASSWORD)
		throw new UnauthorizedError(`bad internal password to access to this url: ${req.url}`, `internal server error`)
}

export default async function tournamentRoutes(app: FastifyInstance) {

	app.post('/join', async (req: FastifyRequest, res: FastifyReply) => {
		try {
			const zod_result = Schema.tournamentIdSchema.safeParse(req.body)
			if (!zod_result.success) {
				const message = zod_result.error.issues[0]?.message || 'Invalid input'
				throw new InputError(message, message)
			}
			let idTournament = zod_result.data.tournamentId

			if (!idTournament)
				throw new InputError(`Empty tournament id`, `tournamentId is empty`)

			const id: number = Number(req.headers.id)
			if (!id)
				throw new ServerError(`cannot parse id, which should not happen`, 500)

			await authUser(id)

			const tournament = tournamentSessions.get(idTournament)
			if (!tournament)
				throw new ConflictError(`there is no tournament with this id`, `this id of tournament doesn't exist`)

			await tournament.addParticipant(id)

			return res.status(200).send()
		} catch (err) {
			if (err instanceof MyError) {
				console.error(err.message)
				console.log(err.message)
				return res.status(err.code).send({ error: err.toSend })
			}
			console.error(err)
			console.log(err)
			return res.status(500).send()
		}
	})

	app.get('/getList', (req: FastifyRequest, res: FastifyReply) => {
		const tournamentList: { participants: number[], maxPlayer: number, status: string }[] = []

		for (const [id, tournament] of tournamentSessions)
			tournamentList.push(tournament.getData())

		return res.status(200).send(tournamentList)
	})


	app.get('/quit/:id', async (req: FastifyRequest, res: FastifyReply) => {
		try {
			const id: number = Number(req.headers.id)
			if (!id)
				throw new ServerError(`cannot parse id, which should not happen`, 500)

			const params = req.params as { id: string }
			const tournamentId = Number(params.id)

			await authUser(id)
			const tournament = tournamentSessions.get(tournamentId)
			if (!tournament)
				throw new InputError(`client try to quit a tournament who doesn't exist`, `this id of tournament doesn't exist`)

			if (!tournament.hasParticipant(id))
				return res.status(409).send({ error: `you are not registered in this tournament` })

			await tournament.quit(id)

			return res.status(200).send()
		} catch (err) {
			if (err instanceof MyError) {
				console.error(err.message)
				return res.status(err.code).send({ error: err.toSend })
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

			await authUser(id)

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
				console.log('launch done')
			})

			return res.status(200).send()
		} catch (err) {
			if (err instanceof MyError) {
				console.error(err.message)
				return res.status(err.code).send({ error: err.toSend })
			}
			console.error(err)
			return res.status(500).send()
		}
	})

	app.get(`/logTournamentStep/:id`, async (req: FastifyRequest, res: FastifyReply) => {
		try {
			const { id } = req.params as { id: string }

			const numericId = Number(id)
			console.log('numereic ID:', numericId)

			if (isNaN(numericId))
				throw new InputError(`the id isn't a number`, `id have to be a number`)

			const tournament = tournamentSessions.get(numericId)
			if (!tournament)
				throw new InputError(`this id of tournament doesn't exist`, `only these id of tournament exist : 4,8,16,32`)

			return res.status(200).send(tournament.sessionData())

		} catch (err) {
			if (err instanceof MyError) {
				console.error(err.message)
				return res.status(err.code).send({ error: err.toSend })
			}
			console.error(err)
			return res.status(500).send()
		}
	})

	app.get(`/userWin/:id`, { preHandler: internalVerification }, async (req: FastifyRequest, res: FastifyReply) => {
		try {
			const { id } = req.params as { id: string }

			let numericId = Number(id)

			if (isNaN(numericId))
				throw new InputError(`the id isn't a number`, `id have to be a number`)

			for (const [tournamentId, tournament] of tournamentSessions) {
				if (tournament.hasPlayer(Math.abs(numericId))) {
					if (numericId > 0)
						await tournament.bracketWon(numericId)
					else
						await tournament.bracketNotPlayed(-numericId)
					return res.status(200).send()
				}
			}
			return res.status(404).send()
		} catch (err) {
			if (err instanceof MyError) {
				console.error(err.message)
				return res.status(err.code).send({ error: err.toSend })
			}
			console.error(err)
			return res.status(500).send()
		}
	})

	app.get(`/test`, async (req: FastifyRequest, res: FastifyReply) => {
		console.log(tournamentSessions.get(4)?.sessionData())

		return res.status(200).send()
	})

}
