import { FastifyReply, FastifyRequest } from 'fastify'
import { OAuth2Client } from 'google-auth-library'
import { Client_db, User } from './User.js'
import { DataBaseError } from './error.js'

const client = new OAuth2Client('562995219569-0icrl4jh4ku3h312qmjm8ek57fqt7fp5.apps.googleusercontent.com')

export async function googleAuth(request: FastifyRequest, reply: FastifyReply): Promise<void> {
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

		if (Client_db.prepare('SELECT * FROM Client WHERE nickName = ?').get(payload.given_name)) {
			let i: number = 1
			while (Client_db.prepare('SELECT * FROM Client WHERE nickName = ?').get(payload.given_name + i))
				i++
			payload.given_name = payload.given_name + i
		}
		if (Client_db.prepare('SELECT * FROM Client WHERE email = ?').get(payload.email))
			console.log('Email Already Register:', payload.email)
		else {
			const res = Client_db.prepare('INSERT INTO Client (nickName, email, password, google_id, pictureProfile) VALUES (?, ?, ?, ?, ?)')
				.run(payload.given_name, payload.email, 'NOTGIVEN', userId, payload.picture)
		}
		const userData = Client_db.prepare('SELECT id FROM Client WHERE email = ?').get(payload.email) as {
			id: number
		} | undefined
		if (!userData)
			throw new DataBaseError('cannot recover id from user connected by google', 'Internal server error', 500)

		const token = User.makeToken(userData.id)
		return reply.send({ token, valid: true, nickName: payload.given_name, avatar: payload.picture })
	} catch (error) {
		console.log('Error verifying Google token:', error)
		reply.status(400).send({ valid: false, error: 'Invalid token' })
	}
}