import { getMatchHistory, MatchResult } from './database.js'
import { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify'
import { generateRoom, startInviteMatch, startTournamentMatch } from './netcode.js'
import { z } from 'zod'
import { INTERNAL_PASSWORD } from './api.js'

const internalVerification = async (req: FastifyRequest) => {
	if (req.headers.authorization !== INTERNAL_PASSWORD)
		throw new Error(`only server can reach this endpoint`)
}

export default async function tdRoutes(fastify: FastifyInstance) {
	fastify.post('/generateRoom', async (req: FastifyRequest, res: FastifyReply) => {
		const roomId = generateRoom()
		return (res.status(200).send({ roomId: roomId }))
	})

	fastify.post('/generateTournamentRoom', async (req: FastifyRequest, res: FastifyReply) => {
		const roomId = generateRoom('tournament')
		return (res.status(200).send({ roomId: roomId }))
	})

	fastify.get('/getMatchHistory', async (req: FastifyRequest, res: FastifyReply) => {
		try {
			const id = Number(req.headers.id)
			const MatchResult: MatchResult[] = getMatchHistory(id)
			return (res.status(200).send(MatchResult))
		} catch (error) {
			console.log(error)
			return (res.status(500).send({ error }))
		}
	})

	fastify.get('/invitationGame/:id', async (req: FastifyRequest, res: FastifyReply) => {
		try {
			const params = req.params as { id: string }
			const oppId = Number(params.id)
			const id = Number(req.headers.id)

			const response = await fetch(`http://social:6500/checkFriend`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'authorization': `${INTERNAL_PASSWORD}`
				},
				body: JSON.stringify({ id1: id, id2: oppId })
			})
			if (!response.ok) {
				res.status(409).send({ message: `this user isn't in your friendlist` })
			}

			const roomID = await startInviteMatch(id, oppId)
			return res.status(200).send({ roomId: roomID })
		} catch (err) {
			console.error(err)
			return res.status(400).send(err)
		}
	})

	fastify.post('/tournamentGame', { preHandler: internalVerification }, async (req: FastifyRequest, res: FastifyReply) => {
		try {

			const ids = z.object({
				id1: z.number(),
				id2: z.number()
			}).parse(req.body)

			const winnerId = await startTournamentMatch(ids.id1, ids.id2)

			return res.status(200).send({ id: winnerId })
		} catch (err) {
			console.error(err)
			return res.status(500).send()
		}
	})
}


