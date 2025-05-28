import { FastifyReply, FastifyRequest } from 'fastify'
import { OAuth2Client } from 'google-auth-library'
import { Client_db, User } from './User.js'
import { ConflictError, DataBaseError } from './error.js'
import { nickNameSchema } from './schema.js'
import { connectedUsers } from './api.js'

const client = new OAuth2Client('562995219569-0icrl4jh4ku3h312qmjm8ek57fqt7fp5.apps.googleusercontent.com')

export async function googleAuth(request: FastifyRequest, reply: FastifyReply) {
	const { credential } = request.body as { credential: string }
	try {
		const ticket = await client.verifyIdToken({
			idToken: credential,
			audience: '562995219569-0icrl4jh4ku3h312qmjm8ek57fqt7fp5.apps.googleusercontent.com',
		})
		const payload = ticket.getPayload()
		const userId = payload?.sub

		if (!payload || !userId) {
			throw new Error('Invalid payload from Google')
		}

		if (Client_db.prepare('SELECT * FROM Client WHERE email = ?').get(payload.email))
			console.log('Email Already Register:', payload.email)
		else {
			const zod_result = nickNameSchema.safeParse({ nickName: payload.given_name })
			if (!zod_result.success)
				payload.given_name = 'googleNickname'

			if (Client_db.prepare('SELECT * FROM Client WHERE nickName = ?').get(payload.given_name)) {
				let i: number = 1
				payload.given_name = payload.given_name + i.toString()
				while (Client_db.prepare('SELECT * FROM Client WHERE nickName = ?').get(payload.given_name))
					i++
			}

			Client_db.prepare('INSERT INTO Client (nickName, email, password, google_id, pictureProfile) VALUES (?, ?, ?, ?, ?)')
				.run(payload.given_name, payload.email, 'NOTGIVEN', userId, payload.picture)
		}


		const userData = Client_db.prepare('SELECT id, nickName, pictureProfile FROM Client WHERE email = ?').get(payload.email) as {
			id: number,
			nickName: string,
			pictureProfile: string
		} | undefined
		if (!userData)
			throw new DataBaseError('cannot recover id from user connected by google', 'Internal server error', 500)

		if (connectedUsers.has(userData.id))
			throw new ConflictError('user try to connect on different sessions', 'you are already connected on an other session')

		const token = User.makeToken(userData.id)

		return reply.code(200).send({
			token,
			valid: true,
			nickName: userData.nickName,
			avatar: userData.pictureProfile
		})
	} catch (error) {
		console.log('Error verifying Google token:', error)
		reply.status(400).send({ valid: false, error: 'Invalid token' })
	}
}